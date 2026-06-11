# JAX LLaVA Project Notes

Follow these notes when editing the JAX LLaVA training/data code in this repo.

## SFT Dataset Loader Notes

- The stage-2/SFT mix includes the LLaVA-OV1.5 image-shuffled data plus VQAv2,
  OKVQA, A-OKVQA, OCRVQA, GQA, TextCaps, Visual Genome QA, Visual Genome
  detection, and RefCOCO.
- Dataset aliases for the four LLaVA-1.5 missing SFT datasets are registered in
  `utils/data_util.py` as `okvqa-train`, `aokvqa-train`, `ocrvqa-train`, and
  `refcoco-train`.
- OKVQA and OCRVQA use the short-answer VQA prompt:
  `Answer the question using a single word or phrase.`
- A-OKVQA uses multiple-choice augmentation in the input pipeline: one grouped
  QA expands to four cyclic choice rotations, and the training target is the
  correct option letter. This matches the LLaVA-1.5 reported ~66K A-OKVQA SFT
  scale better than treating the stored grouped records as direct-answer only.
- OCRVQA is uploaded as the full train split, but SFT mix weights intentionally
  sample it at the LLaVA-1.5 ~80K scale. Do not infer the mix weight from the
  full uploaded 801K QA count unless explicitly changing the recipe.
- RefCOCO records store COCO absolute `xywh` boxes and grouped refs per image.
  The input pipeline expands one phrase-to-box training item per ref and formats
  targets as `<loc....>` tokens with the same `format_detection_prompt()` used
  by RefCOCOg eval.

## LLaVA-1.5 Reproduction Tracking

- Experiment spreadsheet:
  `https://docs.google.com/spreadsheets/d/1FlcygQbGBTqHLJeiKdwxS0nP41SPMJrtX-kCJq8d7SQ/edit?gid=1146448969#gid=1146448969`.
- Rows 146-148 on gid `1146448969` track the first LLaVA-1.5 reproduction
  run. Row 147 is the 7B target line; row 148 is the first init run:
  `CLIP-L @ 336 + gemma3-1B`, blip-3o pretrain, and SFT over LLaVA-OV1.5,
  VQA, GQA, and TextCaps style data.
- Row 148 WandB run: `exalted-haze-9 | Sqa24's workspace | jax-llava`.
- Row 147 target metrics recorded in the sheet:
  MME-P `1511`, VQAv2 `78.5`, TextVQA `58.2`, MMBench `64.3`,
  POPE `84.2`, VStar `~48.7`, OCRBench `29.7`, MMVP `24.7`,
  CountBench `~47.0`, VisWiz `50.0`, SEED-Bench image `66.1`,
  ScienceQA-IMG `66.8`, and GQA `62.0`.
- VStar, OCRBench, MMVP, CountBench, RefCOCOg, and ImageNet KNN are not
  original LLaVA-1.5 paper table metrics. Use them as additional probes only.
- The sheet POPE target `84.2` matches the paper's adversarial POPE split.
  Row 148 records `pope_adversarial_f1_stage2_final=80`, not the macro F1.
  The evaluator also logs macro F1 (`82.58` in the same final eval), but the
  spreadsheet value is adversarial and should be compared to the adversarial
  target.
- First row-148 run data: stage 1 used `blip3o-short`; stage 2 used
  LLaVA-OV1.5, VQAv2, GQA train, TextCaps train, and Visual Genome QA with
  weights `[22, 0.4, 0.94, 0.022, 1.7]`. It did not include OKVQA,
  A-OKVQA, OCRVQA, VG detection, RefCOCO, or ShareGPT text-only.
- Current `configs/remote_run_config.yml` differs from the first run: stage 1
  uses `cc12m`, and stage 2 adds `okvqa-train`, `aokvqa-train`,
  `ocrvqa-train`, `visual-genome-det`, and `refcoco-train` with weights
  `[22, 0.4, 0.009, 0.068, 0.08, 0.94, 0.022, 1.7, 0.86, 0.048]`.
- KNN ImageNet TFDS preprocessing should follow `config.dataset.resize_mode`.
  `stretch` directly resizes to the square canvas; `letterbox` preserves aspect
  ratio and pads, matching the normal train/eval transform family. Do not
  silently use center-crop KNN when the run config says `stretch` or `letterbox`.
- A gist-ready summary of the current image-shuffled LLaVA-OV1.5 dataloader is
  kept at `ideas/llava_ov15_dataloader_gist.md`.
- Parsed v5p-64 throughput for first LLaVA-style curriculum jobs:
  stage 1 median `5.63` steps/s all points (`5.69` filtered >2 steps/s);
  stage 2 median `3.91` steps/s all points (`3.94` filtered >2 steps/s).
  Finished row-148 stage-2 segment `20260531_004908_ggu3cd...` median was
  `3.84` steps/s over 17 logged points.

## Durable Checkpoint Notes

- Final training checkpoints should be mirrored to the same regional bucket
  under `pretrained-ckpts/qiao_zhicheng_hanhong_files/...`; this matches the
  shell helper `ltgcp` and protects checkpoints from ordinary bucket cleanup.
- `load_from_pretrained` params-only restore should first use the normal
  zone-local checkpoint path and then fall back to the same-zone
  `pretrained-ckpts` path. Known `gs://kmh-gcp-*` checkpoint paths are
  rewritten to the current target zone before restore to avoid cross-zone reads.

## HSDP Memory Notes

- `model.prompt_causal` defaults to `True`. This is the standard LLaVA-style
  mask: image prefix tokens are bidirectional, while prompt/text tokens remain
  causal. Set it to `False` only to reproduce the older local behavior where the
  entire image+prompt prefix was bidirectional.
- The projector/connector is its own optimizer group. Use
  `training.connector_learning_rate` (or legacy alias
  `training.projector_learning_rate`) to set its LR separately from the LM main
  group and vision encoder group.
- Eval generation uses task-specific token budgets:
  `eval_tokens_shortqa`, `eval_tokens_mid`, `eval_tokens_ocr`,
  `eval_tokens_refcoco`, `eval_tokens_pixelbench`, and
  `eval_tokens_mmbench`. These samplers still respect `sampling.beam_size`, so a
  deliberate beam-search final eval should set that once rather than patching
  every task.
- `utils/pjit_util.py` treats the last mesh axis as the model axis for `hsdp`.
  Batch/data arrays are sharded only over the preceding data axes; do not shard
  batch over every mesh axis or activations can conflict with model sharding and
  get gathered back to global-batch-sized attention buffers.
- Host-local input/output specs may temporarily fall back to all mesh axes when
  the current TPU logical mesh does not give each process exactly
  `1/process_count` of the data-axis positions. This is only an input boundary
  compatibility fallback for `global_batch/process_count` dataloaders; model
  activations still use the HSDP data-axis/model-axis constraints.
- The HSDP mesh must be process-major with the final model axis host-local when
  possible. `utils/pjit_util.py` now builds such a mesh before falling back to
  JAX's default `create_device_mesh`; this keeps `global_batch / process_count`
  dataloader batches compatible with `make_array_from_process_local_data`. The
  failure signature from the old default mesh was `process data has 32 elements.
  Process addresses 64 elements and global_shape=(256, 64)` on v5p-64.
- Training loss should use `token_xent_loss_from_hidden(...)`, not a full
  `embedder.decode(out)` over `(B, K + T, vocab)`. The chunked loss only scans
  labeled text positions and avoids the large Gemma3 full-vocab logits tensor.
- CLIP/LLaVA and PrefixMAE paths use best-effort `constrain_batch_model(...)`
  constraints around image features, q/k/v, attention maps, and hidden
  activations. Keep these constraints when editing HSDP paths.
- `LlavaGemma.generate_beam_search(...)` is a real cumulative-logprob beam search,
  not a greedy alias. It keeps EOS beams alive, supports `txt_feature_layer > 0`,
  and decodes only the last prompt hidden state during prefill to avoid a
  `[B, prompt_len, vocab]` logits tensor.

## Stateful Dataloader Resume

- `jax_llava` now carries the same full stateful dataloader infrastructure as
  `beifen-Paligemma`. Enable it with `dataset.stateful_dataloader: True`;
  `configs/remote_run_config.yml` enables it by default.
- This requires `torchdata==0.8.0`. Missing `torchdata.stateful_dataloader`
  should fail fast instead of silently falling back to non-exact resume.
- Stateful resume saves sidecars at
  `checkpoint_<step>/dataloader_state/process_<rank>.pkl` and restores WebDataset
  cursors, shuffle buffers, RandomMix state, worker RNGs, and topology metadata.
  It is exact only for the same process count, process-local batch size, worker
  count, prefetch factor, roots/types, mix weights, and seed offset.
- `input_pipeline.create_split` validates that training GCS roots match
  `config.zone` before reading/listing shards. Keep this guard to avoid
  cross-region transfer and accidental loops over nonexistent paths.

## TPU Manager & Job Queue

- **Check job status**: use `tcs` (alias) or `tpu check sqa` to view running/errored/finished jobs, their window IDs, TPU assignments, and log dirs. Do NOT manually parse xibo `data.json` to get job info — `tcs` is the canonical interface.
- **Queue a job**: `qsqa <dir_num>` from the working directory. This stages code (rsync to `/kmh-nfs-ssd-us-mount/staging/sqa/...`), registers with xibo (`tpu set-cur`), and adds to the MONITOR queue.
- **MONITOR queue file**: `/kmh-nfs-ssd-us-mount/code/qiao/work/tpu_manager/queue.json`. MONITOR dispatches pending jobs by calling `tpu run <alias> <user> dir=<dir_no>`, which re-stages from the working directory at dispatch time.
- **Directory numbers**: xibo maps dir numbers to working directory paths in `data.json`. `set-cur` only works for dir 1-100; for dir>100, register directly in `data['users']['sqa']['working_dir']`.
- **Eval-only jobs**: set `eval_only: True` and `load_from: <log_dir_path>` in `remote_run_config.yml`. The code resolves checkpoint paths from log dirs via GCS mirror.
- **Multiple jobs from the same codebase**: since MONITOR re-stages from the live working dir at dispatch time, use separate dir numbers pointing to separate directory copies to avoid config conflicts between queued jobs.
