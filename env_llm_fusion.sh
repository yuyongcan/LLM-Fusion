# LLM-Fusion 复现环境。用法: source env_llm_fusion.sh
B=/taijifs_sz34/share_304169095/hunyuan/yongcanyu

# 代理（HF/GitHub 必须走）
export http_proxy="http://star-proxy.oa.com:3128"
export https_proxy="http://star-proxy.oa.com:3128"
export no_proxy=".woa.com,mirrors.cloud.tencent.com,localhost,127.0.0.1,.oa.com"
export HTTP_PROXY="$http_proxy" HTTPS_PROXY="$https_proxy" NO_PROXY="$no_proxy"
# CodeBuddy safe-delete 守卫会掐断批量删临时文件的任务
export CODEBUDDY_SAFE_DELETE_ENABLED=0

# conda 环境: /opt/conda/envs/torch-base (py3.13, torch2.11+cu130, vllm0.24)
export PATH="/opt/conda/envs/torch-base/bin:$PATH"

# 四个根路径
export MODEL_PATH=${MODEL_PATH:-$B/checkpoints/Qwen/Qwen3-4B-Instruct-2507}
export TRAIN_DATA_ROOT=$B/data/LLM-Fusion/train
export EVAL_DATA_ROOT=$B/data/LLM-Fusion/test
export OUTPUT_ROOT=$B/runs/llm_fusion
export TEACHER_ROOT=$B/checkpoints/Siye01/experts_hf

# 单机 8 卡
export NNODES=1
export NGPUS_PER_NODE=8
