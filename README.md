# LIGM

Long-range Information-Gain Masking (LIGM) is a continued-pretraining method for
ModernBERT. It concentrates masked-language-model supervision on tokens whose
prediction improves when the encoder can use its global attention layers.

The frozen research protocol, thresholds, and stopping rules are documented in
[plan.md](plan.md). Results are reported only after the corresponding gate has
completed; the repository does not treat additional training alone as evidence
that the method works.

## Method

For each document, an EMA teacher first selects 40% whole-word candidates. The
same masked input is evaluated with normal ModernBERT attention and with every
global layer restricted to the 128-token local window:

```text
g_i = log p_global(x_i) - log p_local(x_i)
s_i = max(g_i, 0) * 4 * p_global(x_i) * (1 - p_global(x_i))
```

The student receives standard MLM cross-entropy on 30% of tokens:

- 20% highest-scoring candidate spans;
- 10% uniformly sampled replay spans.

No architecture change, contrastive loss, knowledge-distillation loss, LoRA, or
external model labels are used. The published checkpoint retains ModernBERT's
normal inference architecture.

## Reproducibility

The reference environment is an RTX 3090 with Python 3.11, CUDA 12.4, PyTorch
2.6.0+cu124, Transformers 4.57.6, and FlashAttention 2.8.3. Dependencies are
locked by `uv.lock`.

```bash
uv python install 3.11
uv sync --extra dev --extra gpu --frozen
uv run pytest -q
```

Model and dataset revisions are fixed in the configuration files. Downloads use
`https://hf-mirror.com`; large files are transferred by aria2 and verified by
size and SHA-256 when the Hub exposes a digest.

```bash
export HF_ENDPOINT=https://hf-mirror.com
export HF_HOME=/nvme-data/ligm/hf

uv run ligm-download model answerdotai/ModernBERT-base \
  --revision 8949b909ec900327062f0ebf497f51aef5e6f0c8 \
  --output /nvme-data/ligm/models/ModernBERT-base

uv run ligm-snapshot configs/snapshot.yaml
```

The Ettin snapshot contains six streams and preserves document boundaries.
Documents are assigned to train, validation, and test at 98/1/1 by a stable hash
of the document ID. Data-crop and mask random-number generators are independent,
so random MLM and LIGM consume identical documents and crop offsets.
The realized shard selection and base-model checksums are stored in
[`manifests/`](manifests/).

## Training

Run the short hardware and algorithm checks:

```bash
uv run ligm-integration /nvme-data/ligm/models/ModernBERT-base
uv run ligm-train configs/smoke-8k-random.yaml
uv run ligm-train configs/smoke-ligm.yaml
uv run python scripts/verify_resume.py
```

GLUE, BEIR, MLDR-ID, code retrieval, ColBERT, and broad hyperparameter sweeps are
outside this protocol.

## Online training

The token-matched random curve is trained first. Both methods pause every 25M
tokens and evaluate the same 128 held-out documents with the same mask seed. For
LIGM, training stops immediately when local recovery is more than 0.5 percentage
points below random MLM at the identical token count. The final model is then
restored from the preceding safe checkpoint.

```bash
uv run ligm-train configs/stage2-online-random.yaml
bash scripts/continue_stage2_ligm.sh
```

`continue_stage2_ligm.sh` starts LIGM only after the random run has produced a
valid 1B selection report. It then builds paired document-bootstrap confidence
intervals and evaluates retained 100M/250M/500M/750M/1B checkpoints. Raw
per-document reports remain under each run's `online-evaluation/` directory.

The hard-stop path is covered by a deterministic integration configuration:

```bash
uv run ligm-train configs/smoke-online-random.yaml
uv run ligm-train configs/smoke-online-stop.yaml
```

## Current status

- [x] Revision-pinned model and 1B-token long-document snapshot
- [x] RTX 3090 8K forward, backward, optimizer step, and checkpoint save
- [x] Global/all-local short-context equivalence
- [x] LIGM selection and EMA training smoke test
- [x] Document-level train/validation/test separation
- [x] Online guard and safe-checkpoint rollback verified
- [ ] 1B random reference curve
- [ ] Online LIGM training with a 1B upper bound
- [ ] Hugging Face checkpoint and finalized model card

## Release

After the online run and selection report complete, build, upload, and
verify the model repository from the RTX workstation:

```bash
bash scripts/build_hf_release.sh
bash scripts/publish_hf.sh raincandy-u/ModernBERT-base-LIGM \
  /nvme-data/ligm/release/ModernBERT-base-LIGM
bash scripts/verify_hf_release.sh
```

The release contains standard Transformers weights and tokenizer files, the
model card, resolved configuration, manifests, original first-stage reports,
the complete online curve, and all per-document online evaluation reports.

## License

Apache-2.0. Dataset artifacts are not redistributed by this repository and retain
their original licenses.
