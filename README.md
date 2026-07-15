# LIGM

LIGM (Long-range Information-Gain Masking) is a continued-pretraining method for
ModernBERT. It selects masked-language-model targets by measuring how much the
teacher's probability of the true token drops when ModernBERT's global attention
layers are restricted to the local sliding window.

The research protocol and pre-registered gates are in [plan.md](plan.md). The
repository is under active execution; benchmark results are not claimed until the
gates in that document have been run.

## Method

For a candidate masked token `i`, LIGM computes:

```text
g_i = log p_global(x_i) - log p_local(x_i)
s_i = max(g_i, 0) * 4 * p_global(x_i) * (1 - p_global(x_i))
```

Twenty percent of tokens are selected from the highest-scoring word groups and
ten percent are sampled uniformly from the remaining candidates. The student is
trained with standard MLM cross-entropy. An exponential-moving-average copy of
the student performs selection and receives no gradients.

## Reproducible environment

The training host uses Python 3.11 and `uv`:

```bash
uv python install 3.11
uv sync --extra dev --frozen
uv run pytest
```

Model and dataset artifacts are revision-pinned and downloaded through
`https://hf-mirror.com`. `ligm-download` generates an aria2 control file with
checksums and never falls back silently to another endpoint.

## Commands

```bash
uv run ligm-download model answerdotai/ModernBERT-base \
  --output /nvme-data/ligm/models/ModernBERT-base

uv run ligm-train configs/smoke-random.yaml
uv run ligm-train configs/smoke-ligm.yaml
uv run ligm-report /nvme-data/ligm/runs
```

## Status

- [x] Research protocol frozen
- [ ] 8K integration smoke test
- [ ] Stage 1 random-MLM baseline
- [ ] Stage 1 LIGM run
- [ ] Stage 1 gate decision
- [ ] Final checkpoint and MLDR evaluation

## License

Apache-2.0.
