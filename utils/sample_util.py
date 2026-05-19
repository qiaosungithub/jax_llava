# from absl import logging

import jax
import numpy as np

from utils.logging_util import log_for_0

def generate_fid_samples(params, workdir, config, p_sample_step, run_p_sample_step, prompts: list[str], cfg_scale=1.0, also_idx=False):
  if prompts is None:
    raise ValueError("Prompts must be provided for text-conditional sampling.")

  num_steps = np.ceil(config.eval.num_samples / (config.eval.device_batch_size * jax.device_count())).astype(int)
  per_host_batch = config.eval.device_batch_size * jax.local_device_count()
  global_batch = config.eval.device_batch_size * jax.device_count()
  
  samples_all = []
  indices_all = []

  log_for_0('[Note] the first sample may be significantly slower!')
  for step in range(num_steps):
    sample_idx = step
    global_start = step * global_batch
    host_start = global_start + jax.process_index() * per_host_batch
    host_end = host_start + per_host_batch
    step_prompts = prompts[host_start:host_end]
    if len(step_prompts) < per_host_batch:
      if not prompts:
        raise ValueError("Empty prompt list provided for FID sampling.")
      pad_prompt = step_prompts[-1] if step_prompts else prompts[-1]
      step_prompts = list(step_prompts) + [pad_prompt] * (per_host_batch - len(step_prompts))
    log_for_0(f'Sampling step {step} / {num_steps}...')
    samples = run_p_sample_step(p_sample_step, params, sample_idx=sample_idx, cfg_scale=cfg_scale, prompts=step_prompts)
    samples = jax.device_get(samples) # samples are scattered; then to host cpu
    samples_all.append(samples)
    indices_all.append(np.arange(host_start, host_end))

  samples_all = np.concatenate(samples_all, axis=0) 
  samples_all = samples_all[:config.eval.num_samples]
  if also_idx:
    indices_all = np.concatenate(indices_all, axis=0)
    indices_all = indices_all[:config.eval.num_samples]
    return samples_all, indices_all
  return samples_all
