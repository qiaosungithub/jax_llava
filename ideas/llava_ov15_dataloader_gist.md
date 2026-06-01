# Gist: LLaVA-OV1.5 Image-Shuffled Dataloader

This is the current `jax_llava` dataloader path for the new
`llava-ov-1.5-instruct-image-shuffled-v1` dataset.  The important design point
is that the offline dataset is shuffled at raw image-sample granularity, and
the online dataloader keeps its buffer at the same granularity before expanding
each image into one or more QA turns.

## Dataset Alias

```python
# utils/data_util.py
dataset_name_to_path_dict = {
    # Full build has part-level subdirectories.
    "llava-ov-1.5-instruct-image-shuffled-v1":
        "gs://kmh-gcp-💣/data/llava-ov-1.5-instruct-image-shuffled-v1/part-*/shard-*.tar",
}

dataset_name_to_type_dict = {
    "llava-ov-1.5-instruct-image-shuffled-v1": "llava_ov15",
}
```

## Config

```yaml
# configs/remote_run_config.yml
dataset:
  image_size: 336
  max_txt_len: 2048
  resize_mode: stretch
  num_workers: 16
  prefetch_factor: 4
  item_shuffle_size:
    default: 512
    llava_ov15: 50000

training:
  stage2:
    name: "stage2_visual_instruction_sft"
    dataset:
      max_txt_len: 512
    dataset_items:
      - llava-ov-1.5-instruct-image-shuffled-v1
      - vqav2
      - okvqa-train
      - aokvqa-train
      - ocrvqa-train
      - gqa-train
      - textcaps-train
      - visual-genome
      - visual-genome-det
      - refcoco-train
    mix_weights: [22, 0.4, 0.009, 0.068, 0.08, 0.94, 0.022, 1.7, 0.86, 0.048]
```

## LLaVA Conversation Expansion

```python
# input_pipeline.py
def expand_llava_sample(sample):
    """Expand one raw LLaVA image/conversation record into per-turn QA items."""
    raw = sample.get("json")
    if raw is None:
        return []
    if isinstance(raw, bytes):
        raw = json.loads(raw.decode("utf-8"))

    convs = raw.get("conversations", []) if isinstance(raw, dict) else raw
    if not isinstance(convs, list):
        return []

    img = sample.get("jpg") or sample.get("jpeg") or sample.get("png") or sample.get("webp")
    if img is None:
        return []

    turns = []
    for c in convs:
        if not isinstance(c, dict):
            continue
        speaker = (c.get("from") or c.get("role") or "").strip().lower()
        value = (c.get("value") or c.get("content") or "").replace("<image>", "").strip()
        if not value:
            continue
        if speaker in {"human", "user"}:
            turns.append(("human", value))
        elif speaker in {"gpt", "assistant"}:
            turns.append(("assistant", value))

    out = []
    sample_id = raw.get("id") if isinstance(raw, dict) else None
    for i, (role, answer) in enumerate(turns):
        if role != "assistant":
            continue
        question = ""
        for j in range(i - 1, -1, -1):
            if turns[j][0] == "human":
                question = turns[j][1]
                break
        if not question or not answer:
            continue
        out.append({
            "jpg": img,
            "question": question,
            "aux": {
                "answer": answer,
                "conversation_id": sample_id,
                "turn_idx": i,
            },
        })
    return out
```

## Worker Shard Assignment

```python
# input_pipeline.py
def _worker_seed(base_seed: int, rank: int, data_seed_offset: int = 0) -> int:
    worker = get_worker_info()
    worker_id = 0 if worker is None else int(worker.id)
    return _fold_data_seed(base_seed, data_seed_offset) + int(rank) * 10007 + worker_id * 1009


def _shuffled_worker_urls(root_url, data_seed_offset: int, epoch: int):
    urls = _expand_gcs_glob_if_needed(root_url)
    urls = [urls] if isinstance(urls, str) else list(urls)
    if not urls:
        return []

    worker = get_worker_info()
    worker_id = 0 if worker is None else int(worker.id)
    num_workers = 1 if worker is None else int(worker.num_workers)
    rank = jax.process_index()
    world = jax.process_count()
    stream_id = rank * num_workers + worker_id
    num_streams = max(1, world * num_workers)

    rng = random.Random(_fold_data_seed(7919 + int(epoch), data_seed_offset))
    rng.shuffle(urls)
    selected = urls[stream_id::num_streams]
    if not selected:
        selected = [urls[stream_id % len(urls)]]
    return selected
```

## Raw-Image Buffer Dataloader

```python
# input_pipeline.py
class VQAv2IterableDataset(IterableDataset):
    """IterableDataset over VQA-style WebDataset shards.

    Shuffles raw image samples first, then expands one chosen image sample into
    QA items lazily. Remaining QA items from the same image re-enter the active
    image buffer as one pending entry, which avoids filling the buffer with many
    copies of the same image.
    """

    def __init__(self, root_url, config, tokenizer, num_shards=None, dataset_type="vqav2", data_seed_offset=0):
        expanded_root = _expand_gcs_glob_if_needed(root_url)
        self.root_url = expanded_root.rstrip("/") if isinstance(expanded_root, str) else list(expanded_root)
        self.config = config
        self.tokenizer = tokenizer
        self.transform = get_transforms(
            config.image_size,
            is_train=False,
            resize_mode=_resize_mode_from_config(config),
        )
        self.max_len = config.max_txt_len
        self.num_shards = num_shards or jax.process_count()
        self.shard_rank = jax.process_index()
        self.dataset_type = dataset_type
        self.data_seed_offset = int(data_seed_offset)
        self.mask_token_category_probs = _build_mask_category_distribution(config, dataset_type)

    def __iter__(self):
        rng = random.Random(_worker_seed(2027, self.shard_rank, self.data_seed_offset))
        expand_fn = _EXPAND_FN.get(self.dataset_type, expand_vqa_sample)

        # For llava_ov15 this is 50000 in the current config.
        shuffle_buf = []
        shuffle_sizes = {
            "textvqa": 2000,
            "ocrvqa": 2000,
            "dvqa": 20000,
            "tallyqa": 50000,
        }
        SHUFFLE_SIZE = _item_shuffle_size(
            self.config,
            self.dataset_type,
            shuffle_sizes.get(self.dataset_type, 10000),
        )

        start_skip_max = _stream_start_skip(self.config, self.dataset_type)
        start_skip_remaining = (
            rng.randrange(start_skip_max + 1) if start_skip_max > 0 else 0
        )

        def pop_random_entry():
            idx = rng.randrange(len(shuffle_buf))
            entry = shuffle_buf[idx]
            shuffle_buf[idx] = shuffle_buf[-1]
            shuffle_buf.pop()
            return entry

        def emit_one_from_buffer():
            entry_type, payload = pop_random_entry()
            if entry_type == "pending":
                items = payload
            else:
                items = expand_fn(payload)
                if not items:
                    return None
                rng.shuffle(items)

            chosen = items.pop()
            if items:
                shuffle_buf.append(("pending", items))

            return preprocess_fn(
                chosen,
                self.transform,
                self.tokenizer,
                self.max_len,
                dataset_type=self.dataset_type,
                mask_token_category_probs=self.mask_token_category_probs,
            )

        epoch = 0
        error_handler = make_stop_after_n_errors(_max_wds_errors(self.config))
        while True:
            urls = _shuffled_worker_urls(self.root_url, self.data_seed_offset, epoch)
            epoch += 1
            if not urls:
                continue

            # Manual worker/rank sharding above, so do not let WebDataset split again.
            ds = wds.DataPipeline(
                wds.SimpleShardList(urls),
                wds.tarfile_to_samples(handler=error_handler),
            )

            for sample in ds:
                if sample is None:
                    continue
                if start_skip_remaining > 0:
                    start_skip_remaining -= 1
                    continue

                shuffle_buf.append(("raw", sample))

                while len(shuffle_buf) >= SHUFFLE_SIZE:
                    out = emit_one_from_buffer()
                    if out is not None:
                        yield out
```

## Routing

```python
# input_pipeline.py
_EXPAND_FN = {
    "llava_ov15": expand_llava_sample,
    "llava15": expand_llava_sample,
    # other VQA-style datasets omitted
}


def make_dataset(root, dataset_config, tokenizer, is_train=True, dataset_type="default", data_seed_offset=0):
    if dataset_type in [
        "vqav2", "okvqa", "aokvqa", "ocrvqa", "gqa", "textvqa", "tallyqa",
        "dvqa", "genome", "refcoco", "llava15", "llava_ov15",
    ]:
        return VQAv2IterableDataset(
            root,
            dataset_config,
            tokenizer,
            dataset_type=dataset_type,
            data_seed_offset=data_seed_offset,
        )
```

## Operational Notes

- The full shuffled alias currently exists in `us-east5`; do not queue configs
  using this alias in other zones until the same GCS root exists there.
- `item_shuffle_size.llava_ov15=50000` is intentionally much larger than the
  global batch size. The earlier 1024 buffer did not remove the token-length
  drift.
- The online buffer stores raw image samples and pending per-image QA lists,
  not expanded image copies, so one image with many QA turns still occupies one
  buffer slot until it is selected.
