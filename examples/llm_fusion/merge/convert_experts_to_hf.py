# Copyright 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""FSDP -> HuggingFace conversion that PRESERVES fp32.

The training shards are stored in fp32 (mixed-precision fp32 master weights),
and the RL task vectors (theta_expert - theta_base) are small enough that a
fp32->bf16 round-trip can add quantization noise comparable to the delta
itself, which changes what a task-vector merge produces. Every Merge result in
the paper was computed from fp32 experts.

verl's ``python -m verl.model_merger merge`` used to force bfloat16, which is
why this script exists. It now takes ``--save_dtype`` and defaults to fp32, so
this script is a thin wrapper kept for its stable CLI and for the checkpoints
already converted with it; the subclass below only pins the dtype.

Usage:
    python examples/llm_fusion/merge/convert_experts_to_hf.py \
        --local_dir  <ckpt>/global_step_<N>/actor \
        --target_dir <out>/math
"""

import argparse
import os

from verl.model_merger.base_model_merger import ModelMergerConfig
from verl.model_merger.fsdp_model_merger import FSDPModelMerger


class FP32FSDPModelMerger(FSDPModelMerger):
    """Kept as a named alias: the fp32 behaviour now lives in the base class."""


def main():
    parser = argparse.ArgumentParser("Convert an FSDP actor checkpoint to a fp32 HF model")
    parser.add_argument("--local_dir", required=True, help="Path to the .../global_step_*/actor dir")
    parser.add_argument("--target_dir", required=True, help="Output HF model dir")
    parser.add_argument("--trust-remote-code", action="store_true")
    args = parser.parse_args()

    config = ModelMergerConfig(
        operation="merge",
        backend="fsdp",
        target_dir=args.target_dir,
        local_dir=args.local_dir,
        hf_model_config_path=os.path.join(args.local_dir, "huggingface"),
        trust_remote_code=args.trust_remote_code,
        save_dtype="float32",
    )
    os.makedirs(config.target_dir, exist_ok=True)

    merger = FP32FSDPModelMerger(config)
    merger.merge_and_save()
    merger.cleanup()


if __name__ == "__main__":
    main()
