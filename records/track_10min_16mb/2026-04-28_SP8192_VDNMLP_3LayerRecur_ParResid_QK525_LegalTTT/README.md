# Draft: SP8192 + VDN MLP + 3-Layer Recurrence + Parallel Residuals + Legal TTT

This is an experimental VDN-on-SOTA candidate derived from `2026-04-09_SP8192_3LayerRecur_ParResid_QK525_LegalTTT`. It keeps the SP8192 tokenizer, 3-layer recurrence, parallel residuals, QK gain, MuonEq-R, GPTQ SDClip, Brotli compression, and legal score-first TTT path from that record.

The only intended architectural change is replacing selected transformer MLP sublayers with a PR #818-style HWNODE/VDN MLP. The default candidate is:

```bash
VDN_ENABLED=1 VDN_MODE=hwnode VDN_LAYERS=all VDN_HIDDEN=864 VDN_ORDER=2
```

For same-script baseline reproduction, set `VDN_ENABLED=0`. For recurrent-layer-only ablations, set `VDN_LAYERS=3,4,5`. The older repeated-step VDN path remains available with `VDN_MODE=vdn`.

## Candidate Runs

Production logs are not included yet. The planned promotion gate is a full 8xH100 seed around `<= 1.0793` BPB, then seeds `42, 314, 999` for the final submission.

From this folder, after downloading the SP8192 cached dataset:

```bash
MATCHED_FINEWEB_REPO_ID=kevclark/parameter-golf python3 ../../../data/cached_challenge_fineweb.py --variant sp8192
```

Baseline reproduction from the same wrapper:

```bash
DATA_DIR=../../../data/ SEED=42 VDN_ENABLED=0 QK_GAIN_INIT=5.25 \
  TTT_ENABLED=1 TTT_LR=0.005 TTT_EPOCHS=3 \
  torchrun --standalone --nproc_per_node=8 train_gpt.py
```

Primary VDN candidate A:

```bash
DATA_DIR=../../../data/ SEED=42 VDN_ENABLED=1 VDN_MODE=hwnode VDN_LAYERS=all VDN_HIDDEN=864 VDN_ORDER=2 \
  QK_GAIN_INIT=5.25 TTT_ENABLED=1 TTT_LR=0.005 TTT_EPOCHS=3 \
  torchrun --standalone --nproc_per_node=8 train_gpt.py
```

Candidate B raises matrix precision if candidate A creates artifact headroom:

```bash
MATRIX_BITS=7 MATRIX_CLIP_SIGMAS=12.85 ...
```

Candidate C increases latent width:

```bash
VDN_HIDDEN=960 ...
```

Candidate D limits VDN to the recurrent physical layers:

```bash
VDN_LAYERS=3,4,5 VDN_MODE=hwnode VDN_HIDDEN=864 VDN_ORDER=2 ...
```

Fallback repeated-step VDN ablation:

```bash
VDN_MODE=vdn VDN_DISCRETIZATION=euler VDN_STEPS=8 ...
```

Promotion gates: train under 600s, sliding + TTT eval under 600s, artifact under 16,000,000 bytes, and no change to score-first TTT ordering.

## Prior VDN Context

This should be framed as an ablation on the merged SP8192/legal-TTT lineage. PR #818 is prior HWNODE/VDN background only, not the base record for this submission.
