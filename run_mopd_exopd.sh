#!/usr/bin/env bash
# ExOPD：单教师 MOPD + reward extrapolation（λ>1），4 机 32 卡启动入口。
#
#   bash run_mopd_exopd.sh [hydra overrides...]
#   REWARD_EXTRAPOLATION_LAMBDA=1.5 bash run_mopd_exopd.sh     # 换 λ
#
# 这是 run_mopd_merged_teacher.sh 的**薄包装**，唯一的变量就是 λ。
# 学生侧超参、拓扑、batch、教师模型全部继承，所以两个作业可以逐 step 直接横比 ——
# 这正是本实验要的对照：λ=1（标准 OPD）vs λ=1.25（ExOPD）。
#
# 不复制一份 200 行的理由：train_mopd_single_teacher.sh 当初必须复制，是因为
# hydra 能加 key 不能删 key（往 train_mopd.sh 追加只会变成第 6 个教师）。这里没有
# 那个约束，λ 就是一个普通的环境变量，包装即可。
#
# ---------------------------------------------------------------------------------
# 在做什么
#
# 标准 OPD 等价于一个 dense KL-constrained RL，其中 implicit reward 与 KL 正则被
# 强制 1:1 加权 —— 这就是「OPD 的 student 能贴住 teacher 但很难超过」的原因。
# λ 放开这个比例（arXiv 2602.12125）：
#
#   advantage = (logπ_teacher − logπ_student) + (λ−1)(logπ_teacher − logπ_ref)
#
# λ=1.0 逐位等价于原来的行为；λ=1.25 是论文在所有设定下用的值（也是唯一能让
# 统一 student 全面超过所有域专家的方法）；λ=1.5 论文报告不稳定 —— student 开始
# hack implicit reward，去猛拟合 log ratio 的尖峰。
#
# π_ref 是 verl 自带的 ref worker，加载 actor_rollout_ref.model.path，即**冻结的
# 初始 student**。本场景下 student 与教师同源于一个 base，所以它正是公式要的那个
# 参考点，不需要额外配置任何模型路径。
#
# ⚠️ 代价：ref 模型被 fuse 进 ActorRolloutRefWorker，**不额外占卡**（仍是
#    24 学生 + 8 教师 = 32），但训练池每卡多一个冻结 4B 模型（param_offload=True
#    已开，落 CPU）外加每步一次前向。步时会涨，起训后看 log 实测。
# ---------------------------------------------------------------------------------
set -uo pipefail

B=/taijifs_sz34/share_304169095/hunyuan/yongcanyu
REPO="${B}/code/LLM-Fusion"
cd "${REPO}"

# λ。默认 1.25（论文值）。设成 1.0 就退回基线，但那样应该直接跑
# run_mopd_merged_teacher.sh，本脚本会拒绝（见下面的门禁）。
export REWARD_EXTRAPOLATION_LAMBDA=${REWARD_EXTRAPOLATION_LAMBDA:-1.25}

# 实验名/日志名带上 λ，否则在面板和 ls 里与基线作业分不开。
# 1.25 -> lam1p25（'.' 在实验名里不安全）
LAMBDA_TAG=$(printf '%s' "${REWARD_EXTRAPOLATION_LAMBDA}" | tr '.' 'p')

# ⚠️ 显存：λ≠1 让同一个 worker 里多出一个冻结的 4B 参考模型。
# 基线用的 0.9 是**按没有参考模型时**调的：0.9 × 95.5G = 86G 给 vLLM，加上 actor 的
# FSDP 分片后只剩 5G 左右余量。这正是 HANDOFF §5「step 1~2 正常、第 3 步 OOM」那类
# 问题的配方 —— gpu_memory_utilization 是占**总**显存的比例，不会因为多了个模型而让路。
# 所以这里降到 0.8（多留约 10G）。参考模型虽然 param_offload=True（参数在 CPU），
# 前向时仍要 gather 分片并放激活。
#
# 代价只是 KV cache 变小。HANDOFF §5 实测 0.9→0.7 时步时 288→297 s/it，几乎无损，
# 因为 KV cache 本来就不是瓶颈。**不影响学习动力学，所以与 λ=1 基线的结果仍可横比。**
export ROLLOUT_GPU_MEM_UTIL=${ROLLOUT_GPU_MEM_UTIL:-0.8}

# ================== ExOPD 专属门禁 ==================
# run_mopd_merged_teacher.sh 自己那四道（Ray 卡数 / 教师目录 / tp 整除 / 逐节点
# sandbox）继续生效，这里只补 λ 这一条。
#
# 为什么值得单独查：本项目已经被「命令行里的值 ≠ verl 真正用的值」坑过两次
# （results.md §6.7 教师 tp 静默变拓扑、§6.3 的 ${VAR:-默认} 覆盖时机）。判据一律
# 是**在解析后的 config 上核对**，而不是相信自己传了什么。
echo "[check] λ 与 reference policy ..."
/opt/conda/envs/torch-base/bin/python - "${REWARD_EXTRAPOLATION_LAMBDA}" <<'PYEOF' || exit 1
import sys
lam = float(sys.argv[1])

if lam == 1.0:
    sys.exit(
        "ERROR: λ=1.0 就是标准 OPD，本脚本没有存在意义。\n"
        "       要跑基线请直接用 run_mopd_merged_teacher.sh。"
    )

import os
from hydra import compose, initialize_config_dir

from verl.trainer.ppo.utils import need_reference_policy
from verl.utils import omega_conf_to_dataclass
from verl.workers.config import DistillationLossConfig

cfg_dir = os.path.abspath("verl/trainer/config")
with initialize_config_dir(config_dir=cfg_dir, version_base=None):
    cfg = compose(
        config_name="ppo_trainer",
        overrides=[
            "distillation.enabled=True",
            "distillation.distillation_loss.loss_mode=k1",
            "distillation.distillation_loss.use_policy_gradient=True",
            "distillation.distillation_loss.use_task_rewards=False",
            f"distillation.distillation_loss.reward_extrapolation_lambda={lam}",
            "actor_rollout_ref.actor.use_kl_loss=False",
            "algorithm.use_kl_in_reward=False",
        ],
    )

# 1. 解析后的 config 里 λ 必须真的是我们要的值
parsed = omega_conf_to_dataclass(cfg.distillation.distillation_loss, DistillationLossConfig)
if parsed.reward_extrapolation_lambda != lam:
    sys.exit(f"ERROR: 解析后的 λ 是 {parsed.reward_extrapolation_lambda}，不是 {lam}。")

# 2. ref worker 必须真的会被建出来。不建的话 ref_log_prob 不进 batch，
#    loss 里会 raise —— 但那要等 8 分钟 vLLM 起完才发生。
if not need_reference_policy(cfg):
    sys.exit(
        "ERROR: need_reference_policy() 是 False —— reference 模型不会被建起来，\n"
        "       ref_log_prob 不会进 batch，ExOPD 项无从计算。"
    )

print(f"  λ            : {parsed.reward_extrapolation_lambda}   (解析后)")
print(f"  loss_mode    : {parsed.loss_mode}")
print("  reference    : 会建（verl ref worker = 冻结的初始 student）")
PYEOF

# ================== 交给基线启动器 ==================
# EXPERIMENT_NAME / LOG 在那边都是 ${VAR:-默认}，这里 export 就能盖掉。
# TEACHER_TAG 也让它自己算（默认 merge5 λ=1.0 那个教师）。
export EXPERIMENT_NAME=${EXPERIMENT_NAME:-mopd_exopd_lam${LAMBDA_TAG}_qwen3_4b}
export LOG=${LOG:-${B}/logs/llm_fusion/train_mopd_exopd_lam${LAMBDA_TAG}_$(date +%Y%m%d_%H%M%S).log}

cat <<EOF

======== ExOPD（reward extrapolation）========
  λ            : ${REWARD_EXTRAPOLATION_LAMBDA}   （1.0 = 标准 OPD；论文用 1.25；1.5 报告不稳定）
  对照作业     : run_mopd_merged_teacher.sh（λ=1），学生侧其余参数完全一致
  实验名       : ${EXPERIMENT_NAME}
  日志         : ${LOG}
  额外开销     : 训练池每卡多一个冻结 4B 参考模型 + 每步一次前向（不额外占卡）
  vLLM 显存    : ${ROLLOUT_GPU_MEM_UTIL}（基线 0.9；给参考模型让出余量，见脚本注释）
==============================================

EOF

exec bash "${REPO}/run_mopd_merged_teacher.sh" "$@"
