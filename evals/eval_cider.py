import json
import sys
import jax

from pycocotools.coco import COCO
from pycocoevalcap.eval import COCOEvalCap
from pycocoevalcap.cider.cider import Cider

from utils.logging_util import log_for_0, log_for_all
from utils.eval_io_util import ensure_eval_result_base_dir, eval_result_prefix

import json
import io
# import gcsfs
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader, DistributedSampler

from input_pipeline import preprocess_fn, get_transforms, prepare_batch_data
from functools import partial
import os

from jax.experimental import multihost_utils as mu


import math
from torch.utils.data import Sampler

class DistributedEvalSampler(Sampler):
    """
    Eval-only distributed sampler:
    - no padding
    - no duplication
    - deterministic (shuffle=False)
    Each rank gets indices: rank, rank+R, rank+2R, ...
    """
    def __init__(self, dataset, num_replicas=None, rank=None):
        if num_replicas is None:
            num_replicas = jax.process_count()
        if rank is None:
            rank = jax.process_index()
        self.dataset = dataset
        self.num_replicas = int(num_replicas)
        self.rank = int(rank)
        self.dataset_len = len(dataset)

        # for __len__ only (approx / not used for correctness)
        self.num_samples = (self.dataset_len - self.rank + self.num_replicas - 1) // self.num_replicas

    def __iter__(self):
        return iter(range(self.rank, self.dataset_len, self.num_replicas))

    def __len__(self):
        return self.num_samples



class GCSImageDataset(Dataset):
    def __init__(self, json_path, config, tokenizer):
        with open(json_path, 'r') as f:
            self.data = json.load(f) # 格式: [{"question_id": 1, "image_path": "gs://..."}, ...]
        assert config.zone in ['us-central1', 'us-east5'], f'We only support us-central1 and us-east5 for now!!!'
        self.preprocess_fn = partial(
            preprocess_fn,
            transform=get_transforms(
                config.dataset.image_size,
                is_train=False,
                resize_mode=getattr(config.dataset, "resize_mode", "letterbox"),
            ),
            tokenizer=tokenizer,
            max_len=config.dataset.max_txt_len,
        ) # 根据你的模型输入调整预处理参数

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        img_path = item['image']
        img_path = f'/kmh-nfs-ssd-us-mount/code/hanhong/shared/COCO/val2014/{img_path}'
        q_id = item['question_id']
        
        # 从 GCS 读取图片
        img = Image.open(img_path).convert('RGB')
        
        # 这里可以加入你的预处理 (Resize, ToTensor 等)
        # img = transform(img)
        
        return self.preprocess_fn({"aux": {'question_id': q_id, 'path': img_path}, "jpg": img})

# def collate_fn(batch):
#     # 因为 multi-device compiled calls 通常需要固定的 batch 维度
#     # 这里将 list of dicts 转换为 dict of lists 或 batched tensors
#     return {
#         "aux": [x["aux"] for x in batch],
#         "pixel_values": torch.stack([x["pixel_values"] for x in batch]), # 如果是 Tensor 可以 torch.stack
#     }


def collate_fn(batch):
    """
    加强版 Collate Function:
    1. 过滤 None (处理坏图)
    2. 智能堆叠: 只会对 Tensor 类型的字段进行 Stack
    3. 自动忽略: 字符串(str)、数字(int/float)等非 Tensor 字段，防止报错
    """
    # 1. 过滤 None
    batch = [b for b in batch if b is not None]
    if len(batch) == 0:
        return {}

    collated = {}
    
    # 2. 获取第一个样本的 Keys 作为参考
    first_sample = batch[0]
    
    for key, value in first_sample.items():
        # 3. type check: only stack tensors
        if isinstance(value, torch.Tensor):
            # ensure all samples have this key, prevent KeyError
            try:
                collated[key] = torch.stack([b[key] for b in batch])
            except RuntimeError as e:
                log_for_0(f"⚠️ Stack error for key '{key}': {e}")
                # 可能是 tensor 形状不一致 (比如没 resize 好)，跳过该字段
                raise e
        elif key == 'prefix_len':
            # collated[key] = batch[0][key] # keep the prefix_len
            collated[key] = torch.tensor([b[key] for b in batch], dtype=torch.int32)
        else:
            # 如果是字符串 (如 'txt', '__key__', 'url')，直接忽略，不传给模型
            # pass
            if key == 'aux':
                collated[key] = [b[key] for b in batch] # 保留 aux 供后续使用
            
    return collated

# --- 多卡逻辑 ---



def evaluate_on_coco_caption(res_file, label_file, outfile=None):
    """
    res_file: txt file, each row is [image_key, json format list of captions].
            Each caption is a dict, with fields "caption", "conf".
    label_file: JSON file of ground truth captions in COCO format.
    """
    coco = COCO(label_file)
    print(f'Loading results from {res_file}...', flush=True)
    cocoRes = coco.loadRes(res_file)
    cocoEval = COCOEvalCap(coco, cocoRes)
    cocoEval.scorers = [(Cider(), "CIDEr")]

    # only evaluate CIDEr, ignore all other metrics

    # evaluate on a subset of images by setting
    # cocoEval.params['image_id'] = cocoRes.getImgIds()
    # please remove this line when evaluating the full validation set
    cocoEval.params['image_id'] = cocoRes.getImgIds()

    # evaluate results
    # SPICE will take a few minutes the first time, but speeds up due to caching
    cocoEval.evaluate()
    result = cocoEval.eval
    if not outfile:
        print(result)
    else:
        with open(outfile, 'w') as fp:
            json.dump(result, fp, indent=4)
    return result

def eval_json_path(result_path):
    """
    - results_final.json: format is a list of {"question_id": "495160", "answer": "A baseball game in progress, with a player running the bases."}
    
    - 在 /kmh-nfs-ssd-us-mount/code/hanhong/shared/unify/omni_comprehension_eval_format_files/OMNI_format_coco_caption_test.json 这个文件里面有 question_id和图片path的对应，然后可以直接根据image生成answer，eval只需要 (question_id, answer) pair
    """

    if os.path.isdir(result_path):
        results_final_file = os.path.join(result_path, "results_final.json")
        coco_format_file = os.path.join(result_path, "results_final_coco_format.json")
    else:
        results_final_file = result_path
        stem, ext = os.path.splitext(result_path)
        coco_format_file = f"{stem}_coco_format{ext or '.json'}"

    log_for_0(f"Evaluating 醋 on {results_final_file} with COCO caption...")

    qas = json.load(open(results_final_file, encoding="utf-8"))

    coco_format_json = [{"image_id": qa["question_id"], "caption": qa["answer"]} for qa in qas]
    with open(coco_format_file, "w", encoding="utf-8") as file_obj:
        file_obj.write(json.dumps(coco_format_json))
    acc = evaluate_on_coco_caption(coco_format_file, f"/kmh-nfs-ssd-us-mount/code/hanhong/shared/unify/omni_comprehension_eval_format_files/COCO-Captions_gt_coco_format.json")["CIDEr"] * 100

    log_for_0(f"醋分 (CIDEr score): {acc:.2f}")
    return acc

def eval_cider(p_sample_step, run_p_sample_step, model, tokenizer, params, config):
    # create dataloader of COCO caption test set

    # 假设你在使用多 GPU 环境
    log_for_0("Setting up DataLoader for COCO caption evaluation...")
    dataset = GCSImageDataset("/kmh-nfs-ssd-us-mount/code/hanhong/shared/unify/omni_comprehension_eval_format_files/OMNI_format_coco_caption_test.json", config, tokenizer)
    # sampler = DistributedSampler(dataset, num_replicas=jax.process_count(), rank=jax.process_index(), shuffle=False) # 自动处理多进程数据切分
    sampler = DistributedEvalSampler(dataset, num_replicas=jax.process_count(), rank=jax.process_index())

    loader = DataLoader(
        dataset, 
        batch_size=config.eval.device_batch_size * jax.local_device_count(), 
        sampler=sampler, 
        num_workers=8, # 每个进程开启 8 个线程读取 GCS
        collate_fn=collate_fn
    )
    log_for_0("DataLoader ready. Starting evaluation loop...")

    ALL_OUTS = []
    VIS_IMAGES = []

    for i,batch in enumerate(loader):
        # 1. 准备数据 (假设 compiled sampling 函数接收的是 numpy/jax 数组)
        # images = prepare_for_jax(batch["images"]) 
        assert batch["pixel_values"].shape[0] <= config.eval.device_batch_size * jax.local_device_count(), f"Expected batch size {config.eval.device_batch_size * jax.local_device_count()}, but got {batch['pixel_values'].shape[0]}"
        if len(VIS_IMAGES) < 16:
            VIS_IMAGES.extend(batch["pixel_values"][:16].cpu().numpy())
            # log_for_0(f"Saved {len(VIS_IMAGES)} images for visualization.")
        batch = prepare_batch_data(batch, batch_size=config.eval.device_batch_size * jax.local_device_count()) # 这里你需要实现这个函数来适配你的模型输入格式

        # print batch shape
        log_for_0(str({
            k: (v.shape if hasattr(v, "shape") else type(v)) for k, v in batch.items()
        }))

        input_ids = batch["input_ids"]
        prefix_len = int(batch["prefix_len"][0])
        input_ids = input_ids[:, :, :prefix_len] # remove the padding tokens

        out_strs = run_p_sample_step(p_sample_step, model, tokenizer, params, batch["pixel_values"], input_ids, batch["prefix_len"])
        
        # 2. 收集输出
        for aux, out_str, is_pad in zip(batch["aux"], out_strs, batch["is_pad"].tolist()):
            if not is_pad:
                ALL_OUTS.append({
                    "question_id": aux["question_id"],
                    "answer": out_str,
                })

        if i % 5 == 0:
            log_for_0(f"Batch {i}/{len(dataset)//(config.eval.device_batch_size * jax.device_count())} done. Collected {len(ALL_OUTS)} results so far...")
    mu.sync_global_devices('inference finished') # 等待所有进程都推理完成

    # 3. 保存结果并评测
    base_dir, result_prefix = eval_result_prefix(
        config,
        "cider_cache_dir",
        "/kmh-nfs-ssd-us-mount/data/cached/zhh/coco_caption_eval",
        "cider",
    )
    ensure_eval_result_base_dir(base_dir)

    res_file = f"{result_prefix}.results_{jax.process_index()}.json"
    with open(res_file, 'w', encoding='utf-8') as f:
        json.dump(ALL_OUTS, f, ensure_ascii=False, indent=4)

    mu.sync_global_devices('write json finished') # 等待所有进程都写完结果

    def merge_outputs(prefix):
        filename = f"{prefix}.results_final.json"
        alist = []
        for rank in range(jax.process_count()):
            path = f"{prefix}.results_{rank}.json"
            if not os.path.exists(path):
                raise FileNotFoundError(f"During CIDEr eval, process {rank} results file missing: {path}")
            alist += json.load(open(path, encoding='utf-8'))

        with open(filename, 'w', encoding="utf-8") as f:
            f.write(json.dumps(alist))
        return filename

    
    # TODO: change this to matmul
    if jax.process_index() == 0:
        log_for_0("Merging results and evaluating...")
        merged_file = merge_outputs(result_prefix)
        acc = eval_json_path(merged_file)
    else:
        log_for_all(f"Process {jax.process_index()} waiting for evaluation to finish...")
        acc = 0.0 # this is never used

    mu.sync_global_devices('evaluation finished') # 等待所有进程都评测完成
    return acc, [o['answer'] for o in ALL_OUTS[:16]], VIS_IMAGES[:16]

if __name__ == '__main__':
    eval_json_path('../debug')
