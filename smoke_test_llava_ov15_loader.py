#!/usr/bin/env python3
import argparse
from types import SimpleNamespace

import jax

import input_pipeline


class ConfigNS(SimpleNamespace):
    def get(self, key, default=None):
        return getattr(self, key, default)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--root",
        default="gs://kmh-gcp-us-central1/data/llava-ov-1.5-instruct/configs/*/shard-*.tar",
    )
    parser.add_argument("--batch-size", type=int, default=0)
    parser.add_argument("--image-size", type=int, default=256)
    parser.add_argument("--max-txt-len", type=int, default=64)
    args = parser.parse_args()

    batch_size = args.batch_size or max(1, jax.local_device_count())
    config = ConfigNS(
        model=ConfigNS(lm_backbone_str="gemma3_270M"),
        dataset=ConfigNS(
            root=[args.root],
            types=["llava_ov15"],
            image_size=args.image_size,
            max_txt_len=args.max_txt_len,
            max_len=args.max_txt_len,
            num_workers=0,
            prefetch_factor=2,
            pin_memory=False,
        ),
    )

    loader, tokenizer = input_pipeline.create_split(config, batch_size=batch_size)
    batch = next(iter(loader))
    if not batch:
        raise RuntimeError("Empty batch from LLaVA-OV1.5 loader")

    print("OK: fetched one batch")
    for key, value in batch.items():
        if hasattr(value, "shape"):
            print(f"{key}: shape={tuple(value.shape)} dtype={getattr(value, 'dtype', type(value))}")
        else:
            print(f"{key}: type={type(value)}")

    ids = batch["input_ids"][0].tolist()
    pad_id = tokenizer.special_tokens.PAD
    if pad_id in ids:
        ids = ids[:ids.index(pad_id)]
    print("decoded_sample:")
    print(tokenizer.decode(ids))


if __name__ == "__main__":
    main()
