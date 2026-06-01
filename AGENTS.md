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
