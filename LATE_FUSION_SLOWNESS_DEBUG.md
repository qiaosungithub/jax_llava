# Late-Fusion (txt_feature_layer=13) ~4–5× Slowness — Debug Handoff

> Written 2026-06-13 after a long investigation. Goal of this doc: give a fresh agent
> EVERYTHING (problem, observations, hypotheses, how-to-profile, every experiment + result)
> so they can continue. The headline puzzle is unsolved; see **OPEN** section.

---

## 0. TL;DR

- A jax_llava / beifen-Paligemma **late-fusion** run (`txt_feature_layer=13`, `sharding: hsdp`,
  jit, gemma3_1B, image 336) does **~1.17 steps/s** in stage-1.
- The comparable **txt_feature_layer=0** reference run (wandb `oc5p312v`) does **~5.3 steps/s**
  stage-1 (and **≥3** stage-2), same hardware family / backbone / image size.
- **steps/s is PINNED at ~1.17** across every change to the **dataloader, sharding, and LM dtype**.
  The ONLY knob that ever moved it: removing the SigLIP per-block head-axis sharding constraints
  (0.85 → 1.17). FSDP made it WORSE (0.71).
- => The bottleneck is **NOT** the dataloader, **NOT** the LM (its sharding/dtype don't matter),
  and **NOT** HSDP communication. Strong remaining lead: the **SigLIP encoder / late-fusion
  cross-attention connector** (the only component whose change mattered; txt=0 lacks the fusion),
  or a fixed per-step host/dispatch overhead.
- **The one measurement never obtained: per-op DEVICE TIME (XProf trace).** All inference was
  from HLO op counts/dtypes, which do NOT give wall-clock per op. **Do this first.**

---

## 1. Problem & goal

User observation: the late-fusion stage-1 run is ~4–5× slower than the txt=0 reference, on the
same v5p-64/v6e hardware and gemma3_1B backbone. Find the cause and fix it. (Original suspicion
was the full-state dataloader; that was ruled out — see below.)

Reference (fast): wandb `oc5p312v`, txt_feature_layer=0. Stage-1 logs:
`/kmh-nfs-ssd-us-mount/logs/sqa/paligemma-baseline/20260601_134109_bagh2p_*v5p-64*/output.log`
and `..._20260601_145127_c221zc_*v5p-64*` → steps_per_second median ~5.3 (max ~5.5).

Slow (current late-fusion): window 7574, stagedir `260613015835-whw9yz--code`, txt=13, ~0.85 with
the old SigLIP constraints / ~1.17 after the head-axis fix.

---

## 2. Config & model facts (verified)

- Backbone: **gemma3_1B**, 26 transformer blocks, `num_heads=4`, `num_kv_heads=1`, `head_dim=256`,
  `embed_dim=1152`, sliding-window pattern (5 local + 1 global), `sliding_window_size=512`.
- Vision: CLIP-L14 @ image 336 → **K=576 image tokens**. Stage-1 `max_txt_len=64` (T=64),
  fused seq S=K+T=640. `siglip_from_scratch: true` (SigLIP trained from scratch, full fwd+bwd).
- Late fusion: `txt_feature_layer=13` → text runs through LM blocks 0..12 (prefix), then the fused
  [image(576)+text(64)] runs through blocks 13..25. txt=0 = standard single forward through all 26.
- Stage-1: `freeze_lm: true` → `train.py:_infer_stop_gradient_text_features` (line ~221) auto-returns
  **True**, so the frozen text prefix is forward-only (NOT backpropped) in stage-1. Confirmed.
- `sharding: hsdp`. Mesh from hardcoded `utils/pjit_util.py:TOPOLOGIES` (line 28):
  v6e-16 → `(4,4)` (data axis AXIS_0=4, **model axis AXIS_1=4**); v5p-64 → `(4,4,4)` (model axis=4).
  Model axis = **last** mesh axis (`_model_axis_names`, pjit_util:136). `constrain_batch_model(x, model_dim=-1)`
  shards dim0 over data axes and the chosen dim over the model axis (`_activation_spec`, pjit_util:410).
- curriculum: `llava15_two_stage`. batch_size stage1 256 / stage2 128 (per-device 32).

---

## 3. Environment, how to run, how to profile (operational)

### Launch / queue
- `qsqa <dir_num> [type=v6e-16|v5p-64]` (alias in `~/.bash_aliases`): stages the **current dir**
  to a new stagedir, registers it to `dir_num`, queues via `MONITOR.py`. Needs interactive tmux
  (`tmux display-message`); run inside a tmux session; `yes |` answers the overwrite prompt.
  **Pick a dir_num whose staged dir is ≥1 month old (read the YYMMDD prefix in `tpu ls sqa`).**
- `tpu run <full_tpu_name> dir=<N> sqa [config.X=Y] [tag=...]`: run dir N on a SPECIFIC TPU (auto-zhan,
  mount-check). Config overrides MUST start with `config.` (e.g. `config.dataset.num_workers=32`);
  main.py uses `config_flags.DEFINE_config_file`, so `--config.dataset.num_workers=32` works.
- `python tpu_manager/MONITOR.py queue dir=<N> type=<t>`: re-queue an already-registered dir (no re-stage).
- Stop a run WITHOUT MONITOR auto-resume: `tpu kill-job <window> sqa` (marks status=killed + kills tmux
  window) THEN `tpu kill-remote <full_name> --zone=<z>` (frees accelerators). Both needed.
- Reserve a card: write `sqa_<full_name>_<ts>` into `/kmh-nfs-ssd-us-mount/code/qiao/tpu_lock/`.

### Reliable cards (IMPORTANT — this wasted hours)
- **kaiminghe v6e-16** (us-east5-b / asia-northeast1-b) are RELIABLE: good env, all workers reachable,
  rarely preempt. **Use these for debug.** Mesh (4,4), model axis 4 → still shows HSDP behavior.
- **v5p-64 spot** repeatedly FAILED before the train_step even compiled: SSH-255 ("connect to worker N"),
  `tpu run` hangs, and broken envs (`huggingface-hub==1.5.0` / numpy `_ARRAY_API not found`).
  If you must use v5p-64: **pre-verify** before launching — SSH `--worker=all` lsof (all reachable)
  AND `python -c "import jax,numpy,huggingface_hub"` (good env). Even then it's flaky.
- Find idle: `python tpu_dls/wrap_master.py --cache false | grep IDLE | grep v6e-16`.

### Read results
- `tcs` (= `tpu check sqa`) for windows/status/TPU. `python tpu_manager/see_log.py <window>` → logdir.
- steps/s: `grep -aoE "steps_per_second=[0-9.]+" <logdir>/output.log`. Lines also have
  `curriculum_stage=N` and `step=N`. Drop values <0.9 (warmup/eval/checkpoint outliers).
- "Error"/"Finished" in tcs are often misclassified — verify by reading output.log.

### HLO dump (no UI; what I used)
- Inject at the TOP of `main.py` (before `import jax`):
  ```python
  import os, socket as _s
  os.environ["XLA_FLAGS"] = (os.environ.get("XLA_FLAGS","")
      + " --xla_dump_to=/kmh-nfs-ssd-us-mount/code/qiao/work/<dir>/" + _s.gethostname()
      + " --xla_dump_hlo_as_text").strip()
  ```
  Dumps ~hundreds of `module_*.after_optimizations.txt` per worker host. Train-step module =
  `module_NNNN.jit_train_step.*` (the big one with matmuls + collectives). A saved dump is at
  `work/hlo_dump_v6e/<host>/module_0484.jit_train_step.*after_optimizations.txt`.
- Grep collectives: `all-reduce|all-gather|all-to-all|collective-permute|reduce-scatter`. Matmuls on
  TPU = `custom-call`. Dtype = the `f32[...]`/`bf16[...]` prefix.

### Profiler TRACE (the MISSING piece — DO THIS)
- `jax.profiler.start_trace(logdir)` … run ~10 **steady** steps each `.block_until_ready()` …
  `jax.profiler.stop_trace()`. Open in **XProf**: `xprof --port 8791 <logdir>` (or `tensorboard --logdir`)
  → `trace_viewer` (per-op timeline), `HLO Op Profile` (per-op device time), `Memory Viewer`.
  Needs a browser — the prior agent could not open it; **the user can**. This is the definitive way
  to see WHICH op eats the ~1.17 ceiling (SigLIP vs cross-attn connector vs LM vs fixed overhead).
- Per-op memory also via `jax.profiler.save_device_memory_profile("mem.prof")` (jit-opaque) and
  AOT `compiled = jitted.lower(*args).compile(); compiled.as_text()/cost_analysis()/memory_analysis()`.

---

## 4. Experiments & results (the ablation table)

All stage-1, late-fusion txt=13, measured steady steps/s (>0.9 filter), v6e-16 unless noted.

| # | change | steps/s | takeaway |
|---|---|---|---|
| 0 | original (SigLIP head-axis constraints ON) | **0.85** | baseline slow |
| 1 | SigLIP head-axis constraints OFF (the fix, applied to `jax_llava`) | **1.17** | +37%, the ONLY lever |
| 2 | + per-block `constrain_batch_model` in `_apply_text_feature_layers`/`_apply_lm_from_layer` | 1.17 | no effect |
| 3 | + pure data-parallel (TOPOLOGIES v6 16→`(16,1)`, model axis=1) | 1.18 | no effect — sharding ruled out |
| 4 | + FSDP (`sharding: fsdp`, full param shard) | **0.71** | WORSE — more sharding hurts |
| 5 | + LM forward wrapped in bf16 (`initialize_param_with_dtype(bf16)` in the 2 `_apply_*`) | 1.18 | no effect — LM dtype ruled out |
| ref | txt_feature_layer=0 (oc5p312v) | **~5.3** | the target |

Dataloader (separately, earlier): full-state vs old `RandomMix` ≈ equal (~1.17); num_workers 16→32
no help (slightly worse). **Dataloader fully ruled out.**

Note the bf16 wrap (#5) result is UNVERIFIED at the HLO level — no XLA dump was taken for that run,
so it's possible `initialize_param_with_dtype` did not actually re-cast the **loaded** (checkpoint)
params on read. Worth verifying.

---

## 5. HLO profile of the train-step (module_0484.jit_train_step, v6e-16)

(From the hloprofile run; file under `work/hlo_dump_v6e/<host>/`.)
- ~**3,417 collectives**: all-reduce **1082**, all-gather **950**, all-to-all **626**,
  collective-permute **624**, reduce-scatter **30**.
- Dominant all-gather: `all-gather bf16[64,640,1152] dimensions={2} replica_groups=[4,4]<=[16]`,
  op_name `.../LlavaGemma._apply_*` → the **hidden dim (1152) sharded over the model axis (4)**,
  gathered per layer. SigLIP hidden `bf16[64,577,1024]` all-gather ×124, `[64,577,4096]` (MLP) ×50.
- Matmuls (`custom-call`): **882 total, 590 output f32, 177 bf16**. Attention is float32:
  `f32[64,640,4,640]` (scores) ×1327, plus f32 q/k/v `[64,640,4/1,32]`.
- **BUT**: the sharding ablations (#3 pure-DP, #4 fsdp) prove these collectives are **overlapped /
  not the bottleneck** (collapsing the model axis changed nothing). And the bf16 wrap (#5) shows the
  f32 matmuls aren't the bottleneck either (or the wrap didn't take). So the HLO "looks" comms- and
  f32-heavy, but **device-time** (never measured) must lie elsewhere.

---

## 6. Hypotheses — status

CONFIRMED / DONE:
- **Dataloader is not the cause** (full-state≈old; workers 16→32 no help). Ruled out, multiple ways.
- **SigLIP per-block head-axis constraints were a real bug.** The 8 `constrain_batch_model(q/k/v/attn,
  model_dim=1)` calls in `models/siglip_enc_dec.py` (SelfAttention + CrossAttention) flip the model
  axis onto the head dim every block → reshards. Removing them = +37% (0.85→1.17). **Already applied
  to `jax_llava/models/siglip_enc_dec.py`** (and the disposable test checkouts).

REFUTED:
- HSDP model-axis hidden sharding / the 3400 collectives → NOT the bottleneck (pure-DP=hsdp=1.17).
- FSDP / less model-parallel → WORSE (0.71).
- LM compute dtype (float32 matmuls/attention) → bf16 wrap gave no change (1.18) [UNVERIFIED at HLO level].
- Per-block split sharding constraints → no change.

OPEN (for the next agent — start here):
1. **SigLIP encoder / late-fusion cross-attention connector.** It is the ONLY component whose change
   moved the needle (#0→#1). It runs fwd+bwd, from-scratch, on 576 patches, and the late-fusion
   **CrossAttention** (image attends to layer-13 text features, in `siglip_enc_dec.py`) is something
   **txt=0 does NOT have**. The LM-bf16 wrap (#5) EXCLUDED `vision_encoder`, so SigLIP still runs f32.
   TEST: bf16 the SigLIP encoder + connector; and/or profile its device-time share.
2. **Fixed per-step host/dispatch overhead.** steps/s is suspiciously flat (~1.17) across everything —
   consistent with a non-device bottleneck. Check device-busy fraction in the XProf trace; check for
   per-step host stalls / gaps between steps in log timestamps.
3. **Verify the bf16 wrap actually took effect** (HLO matmul dtypes for the #5 run — no dump taken).
   `initialize_param_with_dtype` may only affect freshly-initialized params, not loaded checkpoint params.
4. **Compare txt=0 (oc5p312v) train-step HLO/trace vs txt=13** to isolate the split's true delta —
   never done (would need oc5p312v's stagedir/config + a dump).

THE decisive missing measurement: **per-op DEVICE TIME via XProf trace** (section 3). Everything above
was inferred from static HLO; we never saw where wall-clock actually goes.

---

## 7. Key file:line references

- `jax_llava/models/llava.py`: `_apply_text_feature_layers` (~279), `_apply_lm_from_layer` (~310),
  `__call__` (~333), `encode_image` (~203), `make_causal_with_prefix_block`, the `constrain_batch_model`
  calls (349,370,404,417).
- `jax_llava/models/siglip_enc_dec.py`: `SelfAttention` / `CrossAttention` (the head-axis constraints —
  now removed in jax_llava; the CrossAttention connector is the late-fusion-specific compute).
- `jax_llava/utils/pjit_util.py`: `TOPOLOGIES` (28), `get_mesh`, `_model_axis_names` (136),
  `_data_axis_names` (142), `_activation_spec` (410), `constrain_batch`/`constrain_batch_model` (432-446).
- `jax_llava/gemma/gm/nn/_transformer.py:234`: `with _dtype_params.initialize_param_with_dtype(self.dtype,
  exclude=['vision_encoder','embedder.mm_input_projection','embedder.mm_soft_embedding_norm','lora'])` —
  the bf16 context LlavaGemma bypasses. `Transformer.dtype` defaults to `jnp.bfloat16` (line 105).
- `jax_llava/train.py:221`: `_infer_stop_gradient_text_features`.
- Reference fast run logs (txt=0, ~5.3): `logs/sqa/paligemma-baseline/20260601_134109_bagh2p_*` and
  `20260601_145127_c221zc_*` (v5p-64). Slow run: window 7574, stagedir `260613015835-whw9yz--code`.

## 8. Disposable test checkouts (under /kmh-nfs-ssd-us-mount/code/qiao/work/)

`jax_llava_hsdp_minfix_sanity` (txt13, SigLIP-fixed, hsdp = the 1.17 baseline),
`jax_llava_splitfix` (+per-block constrain), `jax_llava_fsdp_test` (sharding=fsdp),
`jax_llava_puredp_test` (TOPOLOGIES model-axis=1), `jax_llava_bf16_test` (LM bf16 wrap — see its
`models/llava.py` diff for the exact bf16-context pattern), `jax_llava_hloprofile` (XLA_FLAGS dump).
All safe to delete. HLO dump: `work/hlo_dump_v6e/`.

---

## 9. 2026-06-13 follow-up: txt=0 speed is recoverable; regression is DATA layout in current HSDP

This pass used disposable checkout:
`/kmh-nfs-ssd-us-mount/code/qiao/work/jax_llava_loss_ablation_260613`
registered as xibo dir `92`. Stage1 was shortened to 350 steps, wandb/checkpoints/eval disabled,
dataset `laion-aes`, `max_txt_len=64`, global batch 256, v5p-64.

Previous "profiler" check:
- The available artifact under `work/hlo_dump_v6e/.../module_0484.jit_train_step...` is a full
  `jit_train_step` HLO (forward + loss + backward/update), not a single-forward trace.
- I still did not find an XProf device-time trace; the previous conclusions were from static HLO.

New ablations:
- Current HSDP, `txt_feature_layer=0`, `token_loss_mode=hidden_scan`:
  log `20260613_214349_wofzs9_...llqbt7z88...`, step 100/200/300 =
  **0.524 / 0.726 / 0.728 steps/s**.
- Current HSDP, same but `token_loss_mode=full_decode`:
  log `20260613_215820_s5n0ts_...llqho2ccq...`, step 100/200 =
  **0.528 / 0.730 steps/s**. Loss implementation is not the bottleneck.
- DDP (`--config.sharding=ddp`), same data/model:
  log `20260613_220836_7u2fao_...llql1g3l7...`, step 100/200/300 =
  **1.876 / 5.380 / 6.002 steps/s**. The target `~5` is reachable in current code.
- Added debug sharding mode `hsdp_legacy_data`: HSDP parameter sharding, but DATA inputs sharded over
  **all mesh axes** as in the 2026-06-01 fast checkout. Same data/model:
  log `20260613_222538_skhepr_...llqbt7z88...`, step 100/200/300 =
  **1.370 / 4.539 / 4.839 steps/s**.
- Verification requested by user: `txt_feature_layer=13 + hsdp_legacy_data`, same data/model:
  log `20260613_224732_mcqu4f_...llqbt7z88...`, step 100/200/300 =
  **1.294 / 4.579 / 4.792 steps/s**. This passes the `txt=13` validation: late-fusion itself is
  not what keeps throughput at `~1.17`; the current HSDP DATA layout does.

Interpretation:
- The 2026-06-01 fast run was also `sharding: hsdp`, but its `utils/pjit_util.py` used DATA spec
  `P(tuple(mesh.axis_names))`, i.e. batch was sharded over the model axis too.
- Current HSDP changed DATA/activation layout to use data axes only (`AXIS_0, AXIS_1`) and keep
  `AXIS_2` as a true model axis. That makes stage1 projector pretrain much slower on v5p-64
  (`~0.73`), while DDP and legacy-data HSDP recover old speed.
- This supersedes section 6's earlier "pure-DP=hsdp=1.17" inference for the txt=0 target. The
  difference is not just late-fusion layer 13 or token loss; it is the current HSDP DATA layout.

Code added for debugging:
- `configs/default.py`: `model.token_loss_mode`, `model.token_loss_chunk_size`, and sharding comment.
- `models/llava.py`: optional `token_loss_mode={hidden_scan,full_decode}`; default remains
  `hidden_scan`.
- `utils/pjit_util.py`: non-default `sharding=hsdp_legacy_data` mode. It keeps HSDP param sharding but
  lets DATA/activation batch sharding use all mesh axes, matching the old fast layout.

Open next checks:
- For stage1 on v5p-64, use `sharding=ddp` or `hsdp_legacy_data` if memory allows. `ddp` is fastest
  in this probe; `hsdp_legacy_data` is closest to old HSDP behavior.
- `txt_feature_layer=13 + hsdp_legacy_data` is verified and moves from `~1.17` to `~4.8` steps/s.
- If true HSDP data-axis-only layout is required for memory at longer context/stage2, use XProf on a
  full `jit_train_step` to quantify whether the slow path is dominated by model-axis collectives or
  activation/weight resharding.
