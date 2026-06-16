#!/usr/bin/env python3
"""Populate LLAVA_OV15_CONFIG_SAMPLES / LLAVA_OV15_CONFIG_QUESTIONS in every
repo's utils/llava_ov15_groups.py from tools/count_ov15_questions.py output."""
import json
import re
import sys

COUNTS = sys.argv[1] if len(sys.argv) > 1 else \
    "/kmh-nfs-ssd-us-mount/code/qiao/work/jax_llava_ov15_groupmix/tmp_audit/diag/ov15_counts.json"
REPOS = [
    "/kmh-nfs-ssd-us-mount/code/qiao/work/jax_llava_ov15_groupmix",
    "/kmh-nfs-ssd-us-mount/code/qiao/work/jax_llava",
    "/kmh-nfs-ssd-us-mount/code/qiao/work/beifen-Paligemma",
]

with open(COUNTS) as f:
    counts = json.load(f)


def render(field, key):
    lines = [f"{field} = {{"]
    for cfg in sorted(counts):
        lines.append(f"    {cfg!r}: {int(counts[cfg][key])},")
    lines.append("}")
    return "\n".join(lines)


samples_block = render("LLAVA_OV15_CONFIG_SAMPLES", "images")
questions_block = render("LLAVA_OV15_CONFIG_QUESTIONS", "questions")

for repo in REPOS:
    path = f"{repo}/utils/llava_ov15_groups.py"
    src = open(path).read()
    src = re.sub(r"LLAVA_OV15_CONFIG_SAMPLES = \{[^}]*\}", samples_block, src, count=1)
    src = re.sub(r"LLAVA_OV15_CONFIG_QUESTIONS = \{[^}]*\}", questions_block, src, count=1)
    open(path, "w").write(src)
    print(f"filled {path}: {len(counts)} configs")

tot_img = sum(int(c["images"]) for c in counts.values())
tot_q = sum(int(c["questions"]) for c in counts.values())
print(f"total_images={tot_img} total_questions={tot_q} global_avg_k={tot_q/max(1,tot_img):.3f}")
