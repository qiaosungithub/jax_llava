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
