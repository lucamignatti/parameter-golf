from __future__ import annotations

import copy
import glob
import io
import math
import os
import random
import subprocess
import sys
import time
import uuid
from pathlib import Path

import numpy as np
import sentencepiece as spm
import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch import Tensor, nn
from torch.nn.parallel import DistributedDataParallel as DDP

import train_gpt_hwnode as base


CastedLinear = base.CastedLinear
DistributedTokenLoader = base.DistributedTokenLoader
GPT = base.GPT
Muon = base.Muon
build_sentencepiece_luts = base.build_sentencepiece_luts
dequantize_mixed_int6 = base.dequantize_mixed_int6
eval_val = base.eval_val
load_data_shard = base.load_data_shard
mixed_quantize_int6 = base.mixed_quantize_int6
restore_low_dim_params_to_fp32 = base.restore_low_dim_params_to_fp32


class Hyperparameters(base.Hyperparameters):
    run_id = os.environ.get("RUN_ID", str(uuid.uuid4()))
    val_tokens_limit = int(os.environ.get("VAL_TOKENS_LIMIT", 0))
    num_kv_heads = int(os.environ.get("NUM_KV_HEADS", 4))
    mlp_mult = float(os.environ.get("MLP_MULT", 3.0))
    eval_stride = int(os.environ.get("EVAL_STRIDE", 64))
    bigram_vocab_size = int(os.environ.get("BIGRAM_VOCAB_SIZE", 2048))
    xsa_last_n = int(os.environ.get("XSA_LAST_N", 4))
    late_qat_threshold = float(os.environ.get("LATE_QAT_THRESHOLD", 0.15))
    ngram_enabled = bool(int(os.environ.get("NGRAM_ENABLED", "1")))
    ngram_min_order = int(os.environ.get("NGRAM_MIN_ORDER", 2))
    ngram_max_order = int(os.environ.get("NGRAM_MAX_ORDER", 12))
    ngram_num_buckets = int(os.environ.get("NGRAM_NUM_BUCKETS", 4_194_304))
    ngram_alpha_min = float(os.environ.get("NGRAM_ALPHA_MIN", 0.05))
    ngram_alpha_max = float(os.environ.get("NGRAM_ALPHA_MAX", 0.70))
    ngram_entropy_center = float(os.environ.get("NGRAM_ENTROPY_CENTER", 3.0))
    ngram_entropy_scale = float(os.environ.get("NGRAM_ENTROPY_SCALE", 2.0))
    ngram_min_count = int(os.environ.get("NGRAM_MIN_COUNT", 2))
    hwnode_virtual_layers = int(os.environ.get("HWNODE_VIRTUAL_LAYERS", 2))
    hwnode_term_gates = bool(int(os.environ.get("HWNODE_TERM_GATES", "0")))
    hwnode_state_bias = bool(int(os.environ.get("HWNODE_STATE_BIAS", "0")))
    hwnode_term_gate_init = float(os.environ.get("HWNODE_TERM_GATE_INIT", "1.0"))
    ttt_enabled = bool(int(os.environ.get("TTT_ENABLED", "0")))


def load_validation_tokens(pattern: str, seq_len: int, token_limit: int = 0) -> Tensor:
    files = [Path(p) for p in sorted(glob.glob(pattern))]
    if not files:
        raise FileNotFoundError(f"No files found for pattern: {pattern}")
    tokens = torch.cat([load_data_shard(file) for file in files]).contiguous()
    if token_limit > 0:
        keep = min(tokens.numel(), token_limit + 1)
        tokens = tokens[:keep].contiguous()
    if tokens.numel() < seq_len + 1:
        raise ValueError(f"Need at least {seq_len + 1} tokens, got {tokens.numel()}")
    return tokens


def configure_attention_backend() -> str:
    base._HAS_FA3 = False
    base.flash_attn_3_func = None
    if os.environ.get("USE_FA3", "1") != "1":
        return "sdpa"
    if getattr(torch.version, "hip", None):
        return "sdpa"
    if not torch.cuda.is_available():
        return "sdpa"
    try:
        major, _minor = torch.cuda.get_device_capability()
    except Exception:
        return "sdpa"
    if major < 9:
        return "sdpa"
    try:
        from flash_attn_interface import flash_attn_func as flash_attn_3_func
    except Exception:
        return "sdpa"
    base.flash_attn_3_func = flash_attn_3_func
    base._HAS_FA3 = True
    return "fa3"


def configure_sdp_backends() -> None:
    if not torch.cuda.is_available():
        return
    try:
        from torch.backends.cuda import (
            enable_cudnn_sdp,
            enable_flash_sdp,
            enable_math_sdp,
            enable_mem_efficient_sdp,
        )
    except Exception:
        return
    try:
        enable_cudnn_sdp(False)
        enable_flash_sdp(True)
        enable_mem_efficient_sdp(False)
        enable_math_sdp(False)
    except Exception:
        pass


def maybe_compile(obj, *, dynamic: bool = False, fullgraph: bool = True, enabled: bool = True, log0=None, label: str = ""):
    if not enabled:
        return obj
    try:
        return torch.compile(obj, dynamic=dynamic, fullgraph=fullgraph)
    except Exception as exc:
        if log0 is not None:
            suffix = f" label={label}" if label else ""
            log0(f"compile:disabled{suffix} reason={exc.__class__.__name__}")
        return obj


def make_adamw(params, *, lr: float, betas: tuple[float, float], eps: float, weight_decay: float, fused: bool):
    if fused:
        try:
            return torch.optim.AdamW(
                params,
                lr=lr,
                betas=betas,
                eps=eps,
                weight_decay=weight_decay,
                fused=True,
            )
        except Exception:
            pass
    return torch.optim.AdamW(
        params,
        lr=lr,
        betas=betas,
        eps=eps,
        weight_decay=weight_decay,
    )


def make_adam(params, *, lr: float, betas: tuple[float, float], eps: float, fused: bool):
    if fused:
        try:
            return torch.optim.Adam(
                params,
                lr=lr,
                betas=betas,
                eps=eps,
                fused=True,
            )
        except Exception:
            pass
    return torch.optim.Adam(
        params,
        lr=lr,
        betas=betas,
        eps=eps,
    )


class HWNodeBlockV2(base.HWNodeBlock):
    """Shared-depth Hammerstein-Wiener Neural ODE block.

    This is the intended HWNODE construction:

        h_0 = x
        z_l(0) = phi(W_in h_l)
        z_l(Δt) = exp(A Δt) z_l(0)
        h_{l+1} = psi(W_out z_l(Δt))

    for l = 0, ..., L-1, with the SAME parameters reused at every virtual
    depth step. The Taylor expansion only approximates exp(A Δt); the virtual
    depth itself comes from repeatedly applying this shared HW step.
    """

    def __init__(self, dim: int, state_dim: int, order: int = 3):
        super().__init__(dim, state_dim, order=order)
        self.num_virtual_layers = int(os.environ.get("HWNODE_VIRTUAL_LAYERS", "2"))
        self.use_term_gates = os.environ.get("HWNODE_TERM_GATES", "0") == "1"
        self.use_state_bias = os.environ.get("HWNODE_STATE_BIAS", "0") == "1"
        gate_init = float(os.environ.get("HWNODE_TERM_GATE_INIT", "1.0"))
        if self.use_term_gates:
            self.term_gates = nn.Parameter(torch.full((order,), gate_init, dtype=torch.float32))
        else:
            self.register_parameter("term_gates", None)
        if self.use_state_bias:
            self.state_bias = nn.Parameter(torch.zeros(state_dim, dtype=torch.float32))
        else:
            self.register_parameter("state_bias", None)

    def _exp_A(self, device, dtype):
        use_cache = not self.training and not torch.is_grad_enabled()
        if use_cache and self._cached_exp_A is not None:
            return self._cached_exp_A

        A_normed = self._spectral_norm_A()
        A = (A_normed * self.dt).to(dtype=dtype)
        I = torch.eye(A.shape[0], device=device, dtype=dtype)
        result, Ak = I.clone(), I.clone()
        if self.term_gates is None:
            for k in range(1, self.order + 1):
                Ak = Ak @ A / k
                result = result + Ak
        else:
            gates = self.term_gates.to(dtype=dtype)
            for k in range(1, self.order + 1):
                Ak = Ak @ A / k
                result = result + gates[k - 1] * Ak

        if use_cache:
            self._cached_exp_A = result
        return result

    def _shared_hwnode_step(self, x: Tensor, exp_a: Tensor) -> Tensor:
        """Apply one shared HWNODE step."""
        z = F.leaky_relu(self.fc(x), negative_slope=0.5)
        z = z @ exp_a.T
        if self.state_bias is not None:
            z = z + self.state_bias.to(dtype=z.dtype)[None, None, :]
        y = self.proj(z)
        return F.leaky_relu(y, negative_slope=0.5).square()

    def forward(self, x: Tensor) -> Tensor:
        exp_a = self._exp_A(x.device, x.dtype)
        h = x
        for _ in range(self.num_virtual_layers):
            h = self._shared_hwnode_step(h, exp_a)
        return h


base.HWNodeBlock = HWNodeBlockV2


def make_model(args: Hyperparameters, device: torch.device) -> GPT:
    model = GPT(
        vocab_size=args.vocab_size,
        num_layers=args.num_layers,
        model_dim=args.model_dim,
        num_heads=args.num_heads,
        num_kv_heads=args.num_kv_heads,
        mlp_mult=args.mlp_mult,
        tie_embeddings=args.tie_embeddings,
        tied_embed_init_std=args.tied_embed_init_std,
        logit_softcap=args.logit_softcap,
        rope_base=args.rope_base,
        qk_gain_init=args.qk_gain_init,
        bigram_vocab_size=args.bigram_vocab_size,
        bigram_dim=args.bigram_dim,
        xsa_last_n=args.xsa_last_n,
        rope_dims=args.rope_dims,
        ln_scale=args.ln_scale,
        dtg=args.dtg_enabled,
        ve_enabled=args.ve_enabled,
        ve_dim=args.ve_dim,
        ve_layers=args.ve_layers,
        hwnode_state_dim=args.hwnode_state_dim,
        hwnode_order=args.hwnode_order,
    ).to(device).bfloat16()
    for module in model.modules():
        if isinstance(module, CastedLinear):
            module.float()
    restore_low_dim_params_to_fp32(model)
    return model


def get_gpu_info_text() -> str:
    commands = [
        ["nvidia-smi"],
        ["rocm-smi"],
    ]
    for cmd in commands:
        try:
            proc = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, check=False)
        except FileNotFoundError:
            continue
        out = (proc.stdout or "").strip()
        err = (proc.stderr or "").strip()
        if out:
            return out
        if err:
            return err
    return "gpu_info: unavailable"


_NGRAM_PRIMES = np.array(
    [36313, 27191, 51647, 81929, 131071, 174763, 233017, 286291, 343597, 425143, 524287, 786433],
    dtype=np.int64,
)
_ORDER_MULTS = np.array(
    [0.30, 0.30, 0.97, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0],
    dtype=np.float32,
)


class NgramCache:
    def __init__(self, min_order: int = 2, max_order: int = 12, num_buckets: int = 4_194_304):
        self.min_order = min_order
        self.max_order = max_order
        self.num_orders = max_order - min_order + 1
        self.num_buckets = num_buckets
        self.bucket_mask = np.int64(num_buckets - 1)
        self.ctx_tables = [np.zeros(num_buckets, dtype=np.int32) for _ in range(self.num_orders)]
        self.full_tables = [np.zeros(num_buckets, dtype=np.int32) for _ in range(self.num_orders)]

    def _compute_hashes(self, tokens_np: np.ndarray, start: int, end: int, order_idx: int):
        n = self.min_order + order_idx
        valid_start = max(start, n - 1)
        count = end - valid_start
        if count <= 0:
            return None, None, valid_start
        h = np.zeros(count, dtype=np.int64)
        for k in range(n - 1):
            offset = valid_start - (n - 1) + k
            h ^= tokens_np[offset:offset + count].astype(np.int64) * _NGRAM_PRIMES[k % len(_NGRAM_PRIMES)]
        ctx_h = h & self.bucket_mask
        target_prime = _NGRAM_PRIMES[min(n - 1, len(_NGRAM_PRIMES) - 1)]
        full_h = (h ^ (tokens_np[valid_start:end].astype(np.int64) * target_prime)) & self.bucket_mask
        return ctx_h, full_h, valid_start

    def _bincount_add(self, table: np.ndarray, indices: np.ndarray) -> None:
        counts = np.bincount(indices.astype(np.intp), minlength=self.num_buckets)
        table += counts[:self.num_buckets].astype(table.dtype)

    def build_full(self, tokens_np: np.ndarray) -> None:
        for oi in range(self.num_orders):
            ctx_h, full_h, _ = self._compute_hashes(tokens_np, 0, len(tokens_np), oi)
            if ctx_h is None:
                continue
            self._bincount_add(self.ctx_tables[oi], ctx_h)
            self._bincount_add(self.full_tables[oi], full_h)

    def score_range(self, tokens_np: np.ndarray, start: int, end: int, min_count: int = 2):
        count = end - start
        ngram_prob = np.zeros(count, dtype=np.float32)
        matched_order = np.full(count, -1, dtype=np.int32)
        matched = np.zeros(count, dtype=bool)
        for oi in range(self.num_orders - 1, -1, -1):
            n = self.min_order + oi
            ctx_h, full_h, valid_start = self._compute_hashes(tokens_np, start, end, oi)
            if ctx_h is None:
                continue
            offset = valid_start - start
            ctx_counts = self.ctx_tables[oi][ctx_h]
            full_counts = np.minimum(self.full_tables[oi][full_h], ctx_counts)
            eligible = (ctx_counts >= min_count) & (full_counts > 0) & ~matched[offset:]
            if not np.any(eligible):
                continue
            out_idx = np.where(eligible)[0] + offset
            ngram_prob[out_idx] = full_counts[eligible].astype(np.float32) / np.maximum(ctx_counts[eligible].astype(np.float32), 1.0)
            matched_order[out_idx] = n
            matched[out_idx] = True
        return ngram_prob, matched_order


def get_compiled_forward_logits(model: nn.Module, use_compile: bool, log0) -> callable:
    cached = getattr(model, "_compiled_forward_logits_v2", None)
    if cached is not None:
        return cached
    compiled = maybe_compile(model.forward_logits, dynamic=False, fullgraph=True, enabled=use_compile, log0=log0, label="forward_logits")
    setattr(model, "_compiled_forward_logits_v2", compiled)
    return compiled


def eval_val_sliding_store(
    args: Hyperparameters,
    base_model: nn.Module,
    rank: int,
    world_size: int,
    device: torch.device,
    val_tokens: Tensor,
    base_bytes_lut: Tensor,
    has_leading_space_lut: Tensor,
    is_boundary_token_lut: Tensor,
    stride: int,
    batch_seqs: int = 32,
    eval_seq_len: int | None = None,
    use_compile: bool = True,
    log0=print,
):
    seq_len = eval_seq_len or args.eval_seq_len or args.train_seq_len
    total_tokens = val_tokens.numel() - 1
    window_starts = [ws for ws in range(0, total_tokens, stride) if min(ws + seq_len, total_tokens) - ws >= 1]
    total_windows = len(window_starts)
    my_s = (total_windows * rank) // world_size
    my_e = (total_windows * (rank + 1)) // world_size
    my_windows = window_starts[my_s:my_e]

    model_p_list: list[np.ndarray] = []
    entropy_list: list[np.ndarray] = []
    bytes_list: list[np.ndarray] = []
    position_list: list[np.ndarray] = []
    nll_list: list[np.ndarray] = []

    base_model.eval()
    compiled_logits = get_compiled_forward_logits(base_model, use_compile, log0)
    with torch.inference_mode():
        for bi in range(0, len(my_windows), batch_seqs):
            batch_ws = my_windows[bi:bi + batch_seqs]
            bsz = len(batch_ws)
            x_batch = torch.zeros(bsz, seq_len, dtype=torch.int64, device=device)
            y_batch = torch.zeros(bsz, seq_len, dtype=torch.int64, device=device)
            wlens: list[int] = []
            for i, ws in enumerate(batch_ws):
                end_pos = min(ws + seq_len, total_tokens)
                wlen = end_pos - ws
                wlens.append(wlen)
                chunk = val_tokens[ws:end_pos + 1].to(dtype=torch.int64, device=device)
                x_batch[i, :wlen] = chunk[:-1]
                y_batch[i, :wlen] = chunk[1:]
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                logits = compiled_logits(x_batch)
            logits_f = logits.float()
            log_probs = F.log_softmax(logits_f, dim=-1)
            probs = log_probs.exp()
            nll_all = F.cross_entropy(
                logits_f.reshape(-1, logits_f.size(-1)),
                y_batch.reshape(-1),
                reduction="none",
            ).reshape(bsz, seq_len)
            model_p = probs.gather(2, y_batch.unsqueeze(-1)).squeeze(-1)
            entropy = -(probs * log_probs).sum(dim=-1)
            for i, ws in enumerate(batch_ws):
                wlen = wlens[i]
                s = 0 if ws == 0 else max(wlen - stride, 0)
                positions = np.arange(ws + s + 1, ws + wlen + 1, dtype=np.int64)
                position_list.append(positions)
                model_p_list.append(model_p[i, s:wlen].cpu().numpy().astype(np.float32))
                entropy_list.append(entropy[i, s:wlen].cpu().numpy().astype(np.float32))
                nll_list.append(nll_all[i, s:wlen].cpu().numpy().astype(np.float64))
                tgt = y_batch[i, s:wlen]
                prev = x_batch[i, s:wlen]
                token_bytes = base_bytes_lut[tgt].to(torch.float64)
                token_bytes += (has_leading_space_lut[tgt] & ~is_boundary_token_lut[prev]).to(torch.float64)
                bytes_list.append(token_bytes.cpu().numpy())

    all_positions = np.concatenate(position_list) if position_list else np.array([], dtype=np.int64)
    all_model_p = np.concatenate(model_p_list) if model_p_list else np.array([], dtype=np.float32)
    all_entropy = np.concatenate(entropy_list) if entropy_list else np.array([], dtype=np.float32)
    all_nll = np.concatenate(nll_list) if nll_list else np.array([], dtype=np.float64)
    all_bytes = np.concatenate(bytes_list) if bytes_list else np.array([], dtype=np.float64)

    loss_sum_t = torch.tensor(all_nll.sum(), device=device, dtype=torch.float64)
    token_count_t = torch.tensor(float(len(all_nll)), device=device, dtype=torch.float64)
    byte_count_t = torch.tensor(all_bytes.sum(), device=device, dtype=torch.float64)
    if dist.is_available() and dist.is_initialized():
        dist.all_reduce(loss_sum_t, op=dist.ReduceOp.SUM)
        dist.all_reduce(token_count_t, op=dist.ReduceOp.SUM)
        dist.all_reduce(byte_count_t, op=dist.ReduceOp.SUM)
    val_loss = (loss_sum_t / token_count_t).item()
    val_bpb = val_loss / math.log(2.0) * (token_count_t.item() / byte_count_t.item())
    base_model.train()
    return all_model_p, all_entropy, all_bytes, all_positions, val_loss, val_bpb


def ngram_rescore(
    args: Hyperparameters,
    tokens_np: np.ndarray,
    cache: NgramCache,
    model_p: np.ndarray,
    entropy: np.ndarray,
    token_bytes: np.ndarray,
    positions: np.ndarray,
    rank: int,
    world_size: int,
    device: torch.device,
    log0=print,
) -> tuple[float, float]:
    count = len(positions)
    if count == 0:
        return 0.0, 0.0

    ngram_prob_all, matched_order_all = cache.score_range(tokens_np, 0, len(tokens_np), min_count=args.ngram_min_count)
    ngram_prob = ngram_prob_all[positions]
    matched_order = matched_order_all[positions]
    matched = matched_order >= 0

    alpha = np.zeros(count, dtype=np.float32)
    if np.any(matched):
        order_idx = (matched_order[matched] - cache.min_order).astype(np.int32)
        centers = args.ngram_entropy_center - 0.25 * order_idx.astype(np.float32)
        sig = 1.0 / (1.0 + np.exp(-args.ngram_entropy_scale * (entropy[matched] - centers)))
        raw_alpha = args.ngram_alpha_min + (args.ngram_alpha_max - args.ngram_alpha_min) * sig
        raw_alpha *= _ORDER_MULTS[np.minimum(order_idx, len(_ORDER_MULTS) - 1)]
        alpha[matched] = np.clip(raw_alpha, 0.0, 0.95)

    p_blend = (1.0 - alpha) * model_p + alpha * ngram_prob
    p_blend[~matched] = model_p[~matched]
    p_blend = np.maximum(p_blend, 1e-10)
    nll = -np.log(p_blend).astype(np.float64)

    loss_sum_t = torch.tensor(nll.sum(), device=device, dtype=torch.float64)
    token_count_t = torch.tensor(float(count), device=device, dtype=torch.float64)
    byte_count_t = torch.tensor(token_bytes.sum(), device=device, dtype=torch.float64)
    if dist.is_available() and dist.is_initialized():
        dist.all_reduce(loss_sum_t, op=dist.ReduceOp.SUM)
        dist.all_reduce(token_count_t, op=dist.ReduceOp.SUM)
        dist.all_reduce(byte_count_t, op=dist.ReduceOp.SUM)
    val_loss = (loss_sum_t / token_count_t).item()
    val_bpb = val_loss / math.log(2.0) * (token_count_t.item() / byte_count_t.item())
    matched_count = int(matched.sum())
    if matched_count > 0:
        log0(f"ngram_rescore: matched={matched_count}/{count} ({100.0 * matched_count / max(count, 1):.1f}%) mean_alpha={alpha[matched].mean():.3f}")
    else:
        log0("ngram_rescore: no matches")
    return val_loss, val_bpb


def eval_ngram_two_pass(
    args: Hyperparameters,
    base_model: nn.Module,
    rank: int,
    world_size: int,
    device: torch.device,
    val_tokens: Tensor,
    base_bytes_lut: Tensor,
    has_leading_space_lut: Tensor,
    is_boundary_token_lut: Tensor,
    stride: int,
    batch_seqs: int = 32,
    eval_seq_len: int | None = None,
    use_compile: bool = True,
    log0=print,
):
    t0 = time.perf_counter()
    log0("ngram_two_pass: starting Pass 1 (sliding-window neural eval)")
    model_p, entropy, token_bytes, positions, pass1_loss, pass1_bpb = eval_val_sliding_store(
        args,
        base_model,
        rank,
        world_size,
        device,
        val_tokens,
        base_bytes_lut,
        has_leading_space_lut,
        is_boundary_token_lut,
        stride=stride,
        batch_seqs=batch_seqs,
        eval_seq_len=eval_seq_len,
        use_compile=use_compile,
        log0=log0,
    )
    t_pass1 = time.perf_counter()
    log0(f"ngram_two_pass: Pass 1 done val_bpb={pass1_bpb:.6f} tokens_scored={len(positions)} time={t_pass1 - t0:.1f}s")

    log0(f"ngram_two_pass: building cache orders={args.ngram_min_order}-{args.ngram_max_order} buckets={args.ngram_num_buckets}")
    tokens_np = val_tokens.numpy().astype(np.int16)
    cache = NgramCache(
        min_order=args.ngram_min_order,
        max_order=args.ngram_max_order,
        num_buckets=args.ngram_num_buckets,
    )
    cache.build_full(tokens_np)
    t_cache = time.perf_counter()
    log0(f"ngram_two_pass: cache built in {t_cache - t_pass1:.1f}s")

    log0("ngram_two_pass: starting Pass 2 (n-gram rescore)")
    ng_loss, ng_bpb = ngram_rescore(
        args,
        tokens_np,
        cache,
        model_p,
        entropy,
        token_bytes,
        positions,
        rank,
        world_size,
        device,
        log0=log0,
    )
    t_pass2 = time.perf_counter()
    log0(f"ngram_two_pass: Pass 2 done val_bpb={ng_bpb:.6f} improvement={pass1_bpb - ng_bpb:.6f} time={t_pass2 - t_cache:.1f}s")
    log0(f"ngram_two_pass: total time={t_pass2 - t0:.1f}s")
    return pass1_loss, pass1_bpb, ng_loss, ng_bpb


def main() -> None:
    code = Path(__file__).read_text(encoding="utf-8")
    args = Hyperparameters()
    use_compile = os.environ.get("USE_TORCH_COMPILE", "1") == "1"
    fused_optim = os.environ.get("FUSED_OPTIM", "1") == "1"

    distributed = "RANK" in os.environ and "WORLD_SIZE" in os.environ
    rank = int(os.environ.get("RANK", "0"))
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    if world_size <= 0:
        raise ValueError(f"WORLD_SIZE must be positive, got {world_size}")
    if 8 % world_size != 0:
        raise ValueError(f"WORLD_SIZE={world_size} must divide 8 so grad_accum_steps stays integral")
    grad_accum_steps = 8 // world_size
    grad_scale = 1.0 / grad_accum_steps
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    device = torch.device("cuda", local_rank)
    torch.cuda.set_device(device)
    if distributed:
        dist.init_process_group(backend="nccl", device_id=device)
        dist.barrier()
    master_process = rank == 0

    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    configure_sdp_backends()
    attention_backend = configure_attention_backend()

    logfile = None
    if master_process:
        os.makedirs("logs", exist_ok=True)
        logfile = f"logs/{args.run_id}.txt"
        print(logfile)

    def log0(msg: str, console: bool = True) -> None:
        if not master_process:
            return
        if console:
            print(msg)
        if logfile is not None:
            with open(logfile, "a", encoding="utf-8") as f:
                print(msg, file=f)

    log0(code, console=False)
    log0(f"Python {sys.version} PyTorch {torch.__version__}", console=False)
    log0(get_gpu_info_text(), console=False)
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    if not args.tokenizer_path.endswith(".model"):
        raise ValueError(f"Script only setup for SentencePiece .model file: {args.tokenizer_path}")
    sp = spm.SentencePieceProcessor(model_file=args.tokenizer_path)
    if int(sp.vocab_size()) != args.vocab_size:
        raise ValueError(f"VOCAB_SIZE={args.vocab_size} does not match tokenizer vocab_size={int(sp.vocab_size())}")

    dataset_dir = Path(args.data_path).resolve()
    actual_train_files = len(list(dataset_dir.glob("fineweb_train_*.bin")))
    effective_eval_seq_len = args.eval_seq_len if args.eval_seq_len > 0 else args.train_seq_len
    val_seq_len = max(args.train_seq_len, effective_eval_seq_len)
    val_tokens = load_validation_tokens(args.val_files, val_seq_len, args.val_tokens_limit)
    base_bytes_lut, has_leading_space_lut, is_boundary_token_lut = build_sentencepiece_luts(sp, args.vocab_size, device)

    log0(f"val_bpb:enabled tokenizer_kind=sentencepiece tokenizer_path={args.tokenizer_path}")
    log0(f"train_loader:dataset:{dataset_dir.name} train_shards:{actual_train_files}")
    log0(f"val_loader:shards pattern={args.val_files} tokens:{val_tokens.numel() - 1}")
    log0(f"attention_backend:{attention_backend} compile:{use_compile} fused_optim:{fused_optim}")
    log0(
        f"hwnode:state_dim={args.hwnode_state_dim} order={args.hwnode_order} "
        f"virtual_layers={args.hwnode_virtual_layers} "
        f"term_gates={args.hwnode_term_gates} state_bias={args.hwnode_state_bias}"
    )
    log0(f"ngram:enabled={args.ngram_enabled} orders={args.ngram_min_order}-{args.ngram_max_order} buckets={args.ngram_num_buckets}")

    base.CastedLinear._qat_enabled = args.qat_enabled
    base.CastedLinear._soft_round_alpha = 1.0
    base.CastedLinear._use_soft_round = False
    base.zeropower_via_newtonschulz5 = maybe_compile(
        base.zeropower_via_newtonschulz5,
        dynamic=False,
        fullgraph=True,
        enabled=use_compile,
        log0=log0,
        label="zeropower",
    )

    base_model = make_model(args, device)
    compiled_model = maybe_compile(
        base_model,
        dynamic=False,
        fullgraph=True,
        enabled=use_compile,
        log0=log0,
        label="model",
    )
    model: nn.Module = DDP(compiled_model, device_ids=[local_rank], broadcast_buffers=False) if distributed else compiled_model

    block_named_params = list(base_model.blocks.named_parameters())
    matrix_params = [
        p
        for name, p in block_named_params
        if p.ndim == 2 and not any(pattern in name for pattern in base.CONTROL_TENSOR_NAME_PATTERNS) and "A_weight" not in name
    ]
    scalar_params = [
        p
        for name, p in block_named_params
        if p.ndim < 2 or any(pattern in name for pattern in base.CONTROL_TENSOR_NAME_PATTERNS) or "A_weight" in name
    ]
    if base_model.skip_weights.numel() > 0:
        scalar_params.append(base_model.skip_weights)
    scalar_params.append(base_model.smear.gate)
    if base_model.bigram is not None:
        scalar_params.append(base_model.bigram.scale)

    token_lr = args.tied_embed_lr if args.tie_embeddings else args.embed_lr
    tok_params = [{"params": [base_model.tok_emb.weight], "lr": token_lr, "base_lr": token_lr}]
    if base_model.bigram is not None:
        tok_params.append({"params": [base_model.bigram.embed.weight], "lr": token_lr, "base_lr": token_lr})
        if base_model.bigram.proj is not None:
            matrix_params.append(base_model.bigram.proj.weight)
    if base_model.ve_shared is not None:
        tok_params.append({"params": [base_model.ve_shared.embed.weight], "lr": token_lr, "base_lr": token_lr})
        if base_model.ve_shared.proj is not None:
            matrix_params.append(base_model.ve_shared.proj.weight)
        scalar_params.append(base_model.ve_shared.scale)
        for scale in base_model.ve_layer_scales:
            scalar_params.append(scale)

    optimizer_tok = make_adamw(
        tok_params,
        lr=token_lr,
        betas=(args.beta1, args.beta2),
        eps=args.adam_eps,
        weight_decay=args.adam_wd,
        fused=fused_optim,
    )
    optimizer_muon = Muon(
        matrix_params,
        lr=args.matrix_lr,
        momentum=args.muon_momentum,
        backend_steps=args.muon_backend_steps,
        weight_decay=args.muon_wd,
    )
    for group in optimizer_muon.param_groups:
        group["base_lr"] = args.matrix_lr
    optimizer_scalar = make_adamw(
        [{"params": scalar_params, "lr": args.scalar_lr, "base_lr": args.scalar_lr}],
        lr=args.scalar_lr,
        betas=(args.beta1, args.beta2),
        eps=args.adam_eps,
        weight_decay=args.adam_wd,
        fused=fused_optim,
    )

    optimizers: list[torch.optim.Optimizer] = [optimizer_tok, optimizer_muon, optimizer_scalar]
    if base_model.lm_head is not None:
        optimizer_head = make_adam(
            [{"params": [base_model.lm_head.weight], "lr": args.head_lr, "base_lr": args.head_lr}],
            lr=args.head_lr,
            betas=(args.beta1, args.beta2),
            eps=args.adam_eps,
            fused=fused_optim,
        )
        optimizers.insert(1, optimizer_head)

    n_params = sum(p.numel() for p in base_model.parameters())
    int6_start = args.num_layers - args.int6_last_n
    for i, block in enumerate(base_model.blocks):
        if i >= int6_start:
            for module in block.modules():
                if isinstance(module, CastedLinear):
                    module._clip_range = 31
    if master_process:
        int5_count = sum(1 for module in base_model.modules() if isinstance(module, CastedLinear) and module._clip_range == 15)
        int6_count = sum(1 for module in base_model.modules() if isinstance(module, CastedLinear) and module._clip_range == 31)
        log0(f"mixed_precision: {int5_count} int5 layers, {int6_count} int6 layers (last {args.int6_last_n} blocks)")
    xsa_layers = [i for i, block in enumerate(base_model.blocks) if block.attn.use_xsa]
    log0(f"model_params:{n_params}")
    log0(f"XSA:{xsa_layers} ws:{world_size} gqa:{args.num_heads}/{args.num_kv_heads}")
    log0(f"lr:embed={token_lr} matrix={args.matrix_lr} scalar={args.scalar_lr} batch:{args.train_batch_tokens} wall:{args.max_wallclock_seconds:.0f}s seed:{args.seed}")

    train_loader = DistributedTokenLoader(args.train_files, rank, world_size, device)

    def zero_grad_all() -> None:
        for optimizer in optimizers:
            optimizer.zero_grad(set_to_none=True)

    max_wallclock_ms = 1000.0 * args.max_wallclock_seconds if args.max_wallclock_seconds > 0 else None

    def lr_mul(step: int, elapsed_ms: float) -> float:
        if args.warmdown_iters <= 0:
            return 1.0
        if max_wallclock_ms is None:
            warmdown_start = max(args.iterations - args.warmdown_iters, 0)
            if warmdown_start <= step < args.iterations:
                return max((args.iterations - step) / max(args.warmdown_iters, 1), 0.0)
            return 1.0
        step_ms = elapsed_ms / max(step, 1)
        warmdown_ms = args.warmdown_iters * step_ms
        remaining_ms = max(max_wallclock_ms - elapsed_ms, 0.0)
        return remaining_ms / max(warmdown_ms, 1e-9) if remaining_ms <= warmdown_ms else 1.0

    if args.warmup_steps > 0:
        initial_model_state = {name: tensor.detach().cpu().clone() for name, tensor in base_model.state_dict().items()}
        initial_optimizer_states = [copy.deepcopy(optimizer.state_dict()) for optimizer in optimizers]
        model.train()
        for warmup_step in range(args.warmup_steps):
            zero_grad_all()
            for micro_step in range(grad_accum_steps):
                if distributed:
                    model.require_backward_grad_sync = micro_step == grad_accum_steps - 1
                x, y = train_loader.next_batch(args.train_batch_tokens, args.train_seq_len, grad_accum_steps)
                with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=True):
                    warmup_loss = model(x, y)
                (warmup_loss * grad_scale).backward()
            for optimizer in optimizers:
                optimizer.step()
            zero_grad_all()
            if args.warmup_steps <= 20 or (warmup_step + 1) % 10 == 0 or warmup_step + 1 == args.warmup_steps:
                log0(f"warmup_step:{warmup_step + 1}/{args.warmup_steps}")
        base_model.load_state_dict(initial_model_state, strict=True)
        for optimizer, state in zip(optimizers, initial_optimizer_states, strict=True):
            optimizer.load_state_dict(state)
        zero_grad_all()
        if distributed:
            model.require_backward_grad_sync = True
        train_loader = DistributedTokenLoader(args.train_files, rank, world_size, device)

    ema_state = {name: tensor.detach().float().clone() for name, tensor in base_model.state_dict().items()}
    ema_decay = float(os.environ.get("EMA_DECAY", "0.997"))
    training_time_ms = 0.0
    stop_after_step: int | None = None
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    step = 0

    while True:
        last_step = step == args.iterations or (stop_after_step is not None and step >= stop_after_step)
        should_validate = last_step or (args.val_loss_every > 0 and step % args.val_loss_every == 0)
        if should_validate:
            torch.cuda.synchronize()
            training_time_ms += 1000.0 * (time.perf_counter() - t0)
            val_loss, val_bpb = eval_val(
                args,
                model,
                rank,
                world_size,
                device,
                grad_accum_steps,
                val_tokens,
                base_bytes_lut,
                has_leading_space_lut,
                is_boundary_token_lut,
            )
            log0(f"step:{step}/{args.iterations} val_loss:{val_loss:.4f} val_bpb:{val_bpb:.4f} train_time:{training_time_ms:.0f}ms step_avg:{training_time_ms / max(step, 1):.2f}ms")
            torch.cuda.synchronize()
            t0 = time.perf_counter()
        if last_step:
            if stop_after_step is not None and step < args.iterations:
                log0(f"stopping_early: wallclock_cap train_time:{training_time_ms:.0f}ms step:{step}/{args.iterations}")
            break

        elapsed_ms = training_time_ms + 1000.0 * (time.perf_counter() - t0)
        scale = lr_mul(step, elapsed_ms)
        if base.CastedLinear._use_soft_round and base.CastedLinear._qat_enabled:
            qat_progress = max(0.0, 1.0 - scale / max(args.late_qat_threshold, 0.01))
            base.CastedLinear._soft_round_alpha = 1.0 + 15.0 * qat_progress
        if args.late_qat_threshold > 0 and scale < args.late_qat_threshold and not base.CastedLinear._qat_enabled:
            base.CastedLinear._qat_enabled = True
            base.CastedLinear._use_soft_round = os.environ.get("SOFT_ROUND_QAT", "0") == "1"
            if base.CastedLinear._use_soft_round:
                log0("soft_round_qat:enabled initial_alpha=1.0")
            log0(f"late_qat:enabled step:{step} scale:{scale:.4f}")

        zero_grad_all()
        train_loss = torch.zeros((), device=device)
        for micro_step in range(grad_accum_steps):
            if distributed:
                model.require_backward_grad_sync = micro_step == grad_accum_steps - 1
            x, y = train_loader.next_batch(args.train_batch_tokens, args.train_seq_len, grad_accum_steps)
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=True):
                loss = model(x, y)
            crownq_lambda = float(os.environ.get("CROWN_Q_LAMBDA", "0.01"))
            if base.CastedLinear._qat_enabled and crownq_lambda > 0:
                cq_loss = torch.zeros((), device=device)
                for module in base_model.modules():
                    if isinstance(module, CastedLinear) and module.weight.ndim == 2:
                        weight = module.weight.float()
                        clip_range = float(module._clip_range)
                        row_max = weight.detach().abs().amax(dim=1)
                        delta = row_max / clip_range
                        cq_loss = cq_loss + (weight.pow(2) * delta.pow(2).unsqueeze(1)).mean()
                loss = loss + crownq_lambda * cq_loss / 12.0
            train_loss += loss.detach()
            (loss * grad_scale).backward()
        train_loss /= grad_accum_steps

        frac = min(step / args.muon_momentum_warmup_steps, 1.0) if args.muon_momentum_warmup_steps > 0 else 1.0
        muon_momentum = (1 - frac) * args.muon_momentum_warmup_start + frac * args.muon_momentum
        for group in optimizer_muon.param_groups:
            group["momentum"] = muon_momentum
        for optimizer in optimizers:
            for group in optimizer.param_groups:
                group["lr"] = group["base_lr"] * scale
        if args.grad_clip_norm > 0:
            torch.nn.utils.clip_grad_norm_(base_model.parameters(), args.grad_clip_norm)
        for optimizer in optimizers:
            optimizer.step()
        zero_grad_all()

        with torch.no_grad():
            for name, tensor in base_model.state_dict().items():
                ema_state[name].mul_(ema_decay).add_(tensor.detach().float(), alpha=1.0 - ema_decay)
        step += 1

        approx_training_time_ms = training_time_ms + 1000.0 * (time.perf_counter() - t0)
        should_log_train = args.train_log_every > 0 and (step <= 10 or step % args.train_log_every == 0 or stop_after_step is not None)
        if should_log_train:
            log0(f"step:{step}/{args.iterations} train_loss:{train_loss.item():.4f} train_time:{approx_training_time_ms:.0f}ms step_avg:{approx_training_time_ms / step:.2f}ms")

        reached_cap = max_wallclock_ms is not None and approx_training_time_ms >= max_wallclock_ms
        if distributed and max_wallclock_ms is not None:
            reached_cap_tensor = torch.tensor(int(reached_cap), device=device)
            dist.all_reduce(reached_cap_tensor, op=dist.ReduceOp.MAX)
            reached_cap = bool(reached_cap_tensor.item())
        if stop_after_step is None and reached_cap:
            stop_after_step = step

    log0(f"peak memory allocated: {torch.cuda.max_memory_allocated() // 1024 // 1024} MiB reserved: {torch.cuda.max_memory_reserved() // 1024 // 1024} MiB")
    log0("ema:applying EMA weights")
    current_state = base_model.state_dict()
    ema_sd = {name: tensor.to(dtype=current_state[name].dtype) for name, tensor in ema_state.items()}
    base_model.load_state_dict(ema_sd, strict=True)

    export_sd = base_model.state_dict()
    if master_process:
        torch.save(export_sd, "final_model.pt")
        model_bytes = os.path.getsize("final_model.pt")
        code_bytes = len(code.encode("utf-8"))
        log0(f"Serialized model: {model_bytes} bytes")
        log0(f"Code size: {code_bytes} bytes")

    sd_cpu = {name: tensor.detach().cpu() for name, tensor in export_sd.items()}
    if args.prune_pct > 0:
        for name, tensor in sd_cpu.items():
            if tensor.ndim == 2 and tensor.numel() > 65536:
                thresh = torch.quantile(tensor.abs().float(), args.prune_pct)
                tensor[tensor.abs() < thresh] = 0.0
        log0(f"pruning:{args.prune_pct * 100:.1f}% magnitude pruning applied")

    quant_result, quant_meta = mixed_quantize_int6(sd_cpu, {"mlp", "attn"})
    quant_buf = io.BytesIO()
    torch.save({"w": quant_result, "m": quant_meta}, quant_buf)
    quant_raw = quant_buf.getvalue()
    if base._COMPRESSOR == "zstd":
        quant_blob = base.zstandard.ZstdCompressor(level=22).compress(quant_raw)
    else:
        quant_blob = base.zlib.compress(quant_raw, 9)
    if master_process:
        with open("final_model.int6.ptz", "wb") as f:
            f.write(quant_blob)
        quant_file_bytes = len(quant_blob)
        code_bytes = len(code.encode("utf-8"))
        log0(f"Serialized model int6+{base._COMPRESSOR}: {quant_file_bytes} bytes")
        log0(f"Total submission size int6+{base._COMPRESSOR}: {quant_file_bytes + code_bytes} bytes")

    if distributed:
        dist.barrier()

    with open("final_model.int6.ptz", "rb") as f:
        quant_blob_disk = f.read()
    if base._COMPRESSOR == "zstd":
        raw_state = base.zstandard.ZstdDecompressor().decompress(quant_blob_disk)
    else:
        raw_state = base.zlib.decompress(quant_blob_disk)
    quant_state = torch.load(io.BytesIO(raw_state), map_location="cpu")
    deq_state = dequantize_mixed_int6(quant_state["w"], quant_state["m"], sd_cpu)

    eval_model = make_model(args, device)
    eval_model.load_state_dict(deq_state, strict=True)
    compiled_eval = maybe_compile(
        eval_model,
        dynamic=False,
        fullgraph=True,
        enabled=use_compile,
        log0=log0,
        label="eval_model",
    )

    torch.cuda.synchronize()
    t_qeval = time.perf_counter()
    q_val_loss, q_val_bpb = eval_val(
        args,
        compiled_eval,
        rank,
        world_size,
        device,
        grad_accum_steps,
        val_tokens,
        base_bytes_lut,
        has_leading_space_lut,
        is_boundary_token_lut,
    )
    torch.cuda.synchronize()
    log0(f"final_int6_roundtrip val_loss:{q_val_loss:.4f} val_bpb:{q_val_bpb:.4f} eval_time:{1000.0 * (time.perf_counter() - t_qeval):.0f}ms")
    log0(f"final_int6_roundtrip_exact val_loss:{q_val_loss:.8f} val_bpb:{q_val_bpb:.8f}")

    sw_seq_len = effective_eval_seq_len
    if args.ngram_enabled:
        torch.cuda.synchronize()
        t_ngram = time.perf_counter()
        sw_val_loss, sw_val_bpb, ng_val_loss, ng_val_bpb = eval_ngram_two_pass(
            args,
            eval_model,
            rank,
            world_size,
            device,
            val_tokens,
            base_bytes_lut,
            has_leading_space_lut,
            is_boundary_token_lut,
            stride=args.eval_stride,
            eval_seq_len=sw_seq_len,
            use_compile=use_compile,
            log0=log0,
        )
        torch.cuda.synchronize()
        log0(f"final_int6_sliding_window val_loss:{sw_val_loss:.4f} val_bpb:{sw_val_bpb:.4f} stride:{args.eval_stride} eval_time:{1000.0 * (time.perf_counter() - t_ngram):.0f}ms")
        log0(f"final_int6_sliding_window_exact val_loss:{sw_val_loss:.8f} val_bpb:{sw_val_bpb:.8f}")
        log0(f"ngram_two_pass_exact val_loss:{ng_val_loss:.8f} val_bpb:{ng_val_bpb:.8f}")
        log0(f"final_int8_zlib_roundtrip_exact val_loss:{ng_val_loss:.8f} val_bpb:{ng_val_bpb:.8f}")
    else:
        torch.cuda.synchronize()
        t_slide = time.perf_counter()
        _model_p, _entropy, _token_bytes, _positions, sw_val_loss, sw_val_bpb = eval_val_sliding_store(
            args,
            eval_model,
            rank,
            world_size,
            device,
            val_tokens,
            base_bytes_lut,
            has_leading_space_lut,
            is_boundary_token_lut,
            stride=args.eval_stride,
            eval_seq_len=sw_seq_len,
            use_compile=use_compile,
            log0=log0,
        )
        torch.cuda.synchronize()
        log0(f"final_int6_sliding_window val_loss:{sw_val_loss:.4f} val_bpb:{sw_val_bpb:.4f} stride:{args.eval_stride} eval_time:{1000.0 * (time.perf_counter() - t_slide):.0f}ms")
        log0(f"final_int6_sliding_window_exact val_loss:{sw_val_loss:.8f} val_bpb:{sw_val_bpb:.8f}")

    if distributed:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
