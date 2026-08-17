# Introduction

This repo implements PaliGemma-style Vision-Language Model (VLM) baseline. The next-token-prediction objective is implemented in `models/paligemma.py`.

This branch also incorporates reconstruction for image-side pretraining. A nested mask is optional for recon or understanding loss part.

# Sharding invariant & CE guardrails

Ported from PaliGemma-baseline, where every rule below was learned from a live v5p failure.

**The model axis must stay inside one host** (`utils/pjit_util.py:_process_major_model_axis_mesh`). `mesh_utils.create_device_mesh` optimizes ICI locality and, e.g. on v5p-64 `(2,4,4)`, puts each model group of 4 devices on two hosts. Batches enter `jit` as `P((data...), None)`, so each logical shard is then held by two hosts, and `make_array_from_process_local_data` fills each copy from that host's own dataloader — "replicated" inputs whose replicas differ. This single defect produced: negative CE (labels differ across model ranks, so the label one-hot hits 0 or 2 columns), step-1 `loss=inf`, per-host splits in "replicated" scalar metrics, on-device `scheckne` halts when such inputs met a manual-region collective, and forward activations silently mixing two hosts' samples through every model-axis psum. The mesh is therefore laid out process-major (model axis = one host's local devices) and hard-asserts one host per model group. CPU simulation and single-host TPUs cannot reproduce any of this — validate mesh changes on a real multi-host slice.

Cross-entropy (`models/paligemma.py:token_xent_loss_from_hidden`) chunks over **tokens**, never the vocab axis (no ±inf sentinels), and on multi-host meshes runs its reductions inside a `shard_map` body with per-rank strided chunks; the einsum stays outside the manual region (an einsum inside halts every core — live v5p verdict). `PALIGEMMA_CE_SHARDMAP=0` falls back to the scan path. CE ≥ 0 is structural, and a violation poisons the loss to NaN so a bad run dies at the first bad step.

Guardrail metrics logged every log step:

| metric | healthy | meaning when not |
|---|---|---|
| `nll_min` | ≥ 0 | per-token CE went negative; < −1e−3 also NaN-kills the run |
| `dbg_onehot_count_min/max` | exactly [1, 1] | label one-hot hit 0/2 vocab columns → corrupted label routing |
| `centered_logit_mean` | ~1e−4 | O(1)+ means the centered decode table is bypassed |
| `host_metric_spread` | 0 | hosts disagree on "replicated" scalars → fake replication somewhere |
| `vocab_mean_logit` | drifts freely | pathological only if \|·\| reaches the softcap |

Full investigation record: `DEBUG_LOG_negative_vlm_loss.md` in the parent work directory.

# Evaluation dependencies

Install the exact evaluation dependencies before running the production final
evaluation:

```bash
python -m pip install --force-reinstall --no-deps -r requirements-eval.txt
```

The force reinstall matters because the internal packages keep a stable
package version across Git commits; `--no-deps` preserves this repo's pinned
JAX/NumPy environment.

MMStar is dispatched through the pinned `one-benchmark-suite` evaluator, which
owns its committed-artifact validation, prompt contract, and official scoring;
`one-dataset-suite` is pinned alongside it for the artifact contract. The
production config resolves `mmstar_root` to the current zone's
`gs://kmh-gcp-${ZONE}/data/vlm_eval_benchmarks/mmstar` mirror and includes
`mmstar` in `stage2.final_eval_tasks`.
