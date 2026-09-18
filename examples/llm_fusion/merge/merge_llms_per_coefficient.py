#!/usr/bin/env python3
"""Task Arithmetic with a *per-expert* scaling coefficient.

The vendored MergeLM only takes one float:

    merged = base + scaling_coefficient * sum_i (expert_i - base)

and asserts it is a float, so it cannot express "give the if expert a bigger
step than the rest". That matters here because the coefficient sweep in
outputs_llm_fusion/merge_4b does not peak at the same lambda for every
benchmark: ifbench keeps improving up to 1.2 while aime25/gpqa/lcb_v5 peak at
1.0 and then fall.

This script computes the per-expert generalisation:

    merged = base + sum_i lambda_i * (expert_i - base)

Everything else matches MergeLM's task arithmetic exactly: fp32 on CPU,
`named_parameters()` (buffers such as rotary inv_freq are left as the base's),
no masking, no rescaling.

Memory: MergeLM holds the base and all experts at once. Here the experts are
consumed one at a time and freed, so peak RAM is base + accumulator + one
expert (~48 GB at 4B) instead of ~96 GB.

Usage (PYTHONPATH must include third_party/MergeLM for utils.utils):
    python3 merge_llms_per_coefficient.py \
        --pretrained_model_name <base> \
        --models_to_merge <a>,<b>,... \
        --scaling_coefficients 1.0,1.2,... \
        --save_model_path ./merged_models/<name> \
        --save_dtype bfloat16
"""

import argparse
import gc
import logging
import os
import sys
import time

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

# utils.utils lives in the vendored MergeLM; run_merge.sh puts it on PYTHONPATH.
from utils.utils import get_param_names_to_merge

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def parse_args():
    parser = argparse.ArgumentParser("Task arithmetic with per-expert scaling coefficients")
    parser.add_argument("--pretrained_model_name", type=str, required=True)
    parser.add_argument("--models_to_merge", type=str, required=True,
                        help="comma-separated expert directories")
    parser.add_argument("--scaling_coefficients", type=str, required=True,
                        help="comma-separated floats, one per model, same order as --models_to_merge")
    parser.add_argument("--save_model_path", type=str, required=True)
    parser.add_argument("--save_dtype", type=str, default="bfloat16", choices=["float32", "bfloat16"],
                        help="on-disk dtype; the merge itself is always computed in fp32")
    return parser.parse_args()


def main():
    args = parse_args()
    logger.info(f"configuration is {args}")

    expert_paths = [p for p in args.models_to_merge.split(",") if p]
    coefficients = [float(x) for x in args.scaling_coefficients.split(",") if x]

    if len(expert_paths) != len(coefficients):
        logger.error(f"got {len(expert_paths)} models but {len(coefficients)} coefficients")
        sys.exit(1)
    if not expert_paths:
        logger.error("nothing to merge")
        sys.exit(1)

    for path in expert_paths:
        if not os.path.isfile(os.path.join(path, "config.json")):
            logger.error(f"{path} is not an HF model dir (no config.json)")
            sys.exit(1)

    logger.info("loading base model (fp32, cpu)")
    base_model = AutoModelForCausalLM.from_pretrained(
        pretrained_model_name_or_path=args.pretrained_model_name, device_map="cpu")
    base_params = {name: param for name, param in base_model.named_parameters()}
    param_names_to_merge = get_param_names_to_merge(
        input_param_names=list(base_params.keys()), exclude_param_names_regex=[])
    logger.info(f"{len(param_names_to_merge)} parameter tensors will be merged")

    # Accumulate sum_i lambda_i * (expert_i - base) on the fly so only one
    # expert is resident at a time.
    accumulator = {name: torch.zeros_like(base_params[name]) for name in param_names_to_merge}

    for expert_path, coefficient in zip(expert_paths, coefficients):
        logger.info(f"loading expert {expert_path} with lambda={coefficient}")
        expert_model = AutoModelForCausalLM.from_pretrained(
            pretrained_model_name_or_path=expert_path, device_map="cpu")
        expert_params = dict(expert_model.named_parameters())

        missing = [n for n in param_names_to_merge if n not in expert_params]
        if missing:
            logger.error(f"{expert_path} is missing {len(missing)} parameters, e.g. {missing[:3]}")
            sys.exit(1)

        with torch.no_grad():
            for name in param_names_to_merge:
                accumulator[name].add_((expert_params[name] - base_params[name]), alpha=coefficient)

        del expert_model, expert_params
        gc.collect()
        logger.info(f"  done, resident RSS now {_rss_gib():.1f} GiB")

    with torch.no_grad():
        for name in param_names_to_merge:
            base_params[name].data.copy_(base_params[name] + accumulator[name])
    del accumulator
    gc.collect()

    if args.save_dtype == "bfloat16":
        logger.info("casting merged model to bfloat16 before saving")
        base_model = base_model.to(torch.bfloat16)

    logger.info(f"saving models at {args.save_model_path}...")
    os.makedirs(args.save_model_path, exist_ok=True)
    base_model.save_pretrained(save_directory=args.save_model_path)
    tokenizer = AutoTokenizer.from_pretrained(pretrained_model_name_or_path=args.pretrained_model_name)
    tokenizer.save_pretrained(save_directory=args.save_model_path)
    logger.info(f"models are saved at {args.save_model_path}")


def _rss_gib():
    try:
        with open("/proc/self/statm") as handle:
            pages = int(handle.read().split()[1])
        return pages * os.sysconf("SC_PAGE_SIZE") / (1024 ** 3)
    except Exception:
        return -1.0


if __name__ == "__main__":
    start = time.time()
    main()
    logger.info(f"finished in {(time.time() - start) / 60:.1f} min")
