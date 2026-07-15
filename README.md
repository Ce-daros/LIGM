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

Run the paired first-stage experiments and focused evaluation:

```bash
bash scripts/run_stage1.sh
bash scripts/evaluate_stage1.sh
# Only after mechanism-gate.json reports passed:
bash scripts/run_stage1_conditionals.sh
```

The stage-one gate uses only:

- synthetic distance-controlled recovery and score correlation;
- held-out natural-document local and long-distance repetition recovery;
- a fixed 250K MS MARCO hard-negative probe on MLDR-English dev if the mechanism
  checks pass.

Download the retrieval data only after the mechanism checks pass:

```bash
bash scripts/download_retrieval.sh
uv run ligm-retrieval train configs/retrieval-probe.yaml MODEL_PATH --output OUTPUT_DIR
uv run ligm-retrieval evaluate configs/retrieval-probe.yaml RETRIEVER_PATH \
  --output EVALUATION_DIR
```

GLUE, BEIR, MLDR-ID, code retrieval, ColBERT, and broad hyperparameter sweeps are
outside this protocol.

## Current status

- [x] Revision-pinned model and 1B-token long-document snapshot
- [x] RTX 3090 8K forward, backward, optimizer step, and checkpoint save
- [x] Global/all-local short-context equivalence
- [x] LIGM selection and EMA training smoke test
- [x] Document-level train/validation/test separation
- [ ] Stage-one random-MLM baseline
- [ ] Stage-one LIGM run
- [ ] Stage-one gate decision
- [ ] Conditional stage-two training and final checkpoint
- [ ] Hugging Face model repository and finalized model card

Raw JSONL metrics, resolved configurations, gate reports, and final benchmark
outputs will be committed after each run completes.

## License

Apache-2.0. Dataset artifacts are not redistributed by this repository and retain
their original licenses.
