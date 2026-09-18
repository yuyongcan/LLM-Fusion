#!/usr/bin/env bash
# 方案 B：teacher 在训练进程内用 HF 加载（不用 vLLM 教师池），4 机 32 卡全做训练。
#
#   bash run_mopd_hf_teacher.sh [hydra overrides...]
#
# 这是 run_mopd_merged_teacher.sh 的**薄包装**，和 run_mopd_exopd.sh 同一个套路：
# 唯一改的是 teacher 跑在哪，学生侧超参、batch、教师权重全部继承，所以可以和
# 基线作业逐 step 横比。
#
# ---------------------------------------------------------------------------------
# 在做什么
#
# 基线把 teacher 放在独立的 vLLM 池里，rollout 阶段逐样本用 prompt_logprobs 取
# log prob。但 teacher 从不生成 —— 它只对学生已经产出的序列做一次 prefill ——
# 所以 vLLM 的看家本领（continuous batching / PagedAttention，全是 decode 阶段的
# 优化）在这里用不上，却要占 8 张卡，而且这 8 张卡不算在训练 world size 里。
#
# in_process 模式把 teacher 换成一个冻结的 HF 模型，fuse 进 ActorRolloutRefWorker
# ——和 ref policy 完全一样的方式，不占独立池 —— 每步做一次批量前向。
#
# 收益：
#   1. 8 张卡回到训练池（24 → 32），拓扑从 3 机训练 + 1 机教师变成 4 机全训练
#   2. 摆脱 vLLM 教师的启动脆弱性（HANDOFF §5 那个 EADDRINUSE 端口竞争）
#   3. 解开 temperature == 1.0 的限制（那是 vLLM 的 prompt_logprobs 限制，
#      HF 前向可以任意缩放 logits）。⚠️ 但学生侧 rollout.temperature=1 仍须钉住，
#      那是 on-policy 正确性要求，和教师那条限制不是一回事，别混为一谈。
#   4. teacher 权重进到训练进程 —— 这是后续做 representation 蒸馏的前提
#
# 代价：每步多一次 32 卡上的 teacher 前向。**步时不一定更快**：基线是 24 卡训练 +
# 教师池自由并发，这里是 32 卡训练 + 一次串行前向。孰优孰劣不是先验的，起训后
# 看 log 实测再判断这条路值不值得。
#
# ⚠️ 本脚本目前**不加 reward extrapolation**（λ 保持 1.0）。第一次跑方案 B 应该是
#    等价性验证：只动 teacher 后端这一个变量，才能和现有单教师 MOPD 直接对比，
#    确认迁移没有改变数值。确认之后再叠 λ。
# ---------------------------------------------------------------------------------
set -uo pipefail

B=/taijifs_sz34/share_304169095/hunyuan/yongcanyu
REPO="${B}/code/LLM-Fusion"
cd "${REPO}"

export DISTILLATION_MODE=in_process

# 教师池收掉，训练池吃下这 8 张卡。
# NNODES_OVERRIDE 是基线脚本读的那个名字（它自己会 source env_llm_fusion.sh 之后
# 写回 NNODES，见那边的注释）；TEACHER_POOL=0 让 WORLD_SIZE 回到 32 = 4×8。
export NNODES_OVERRIDE=${NNODES_OVERRIDE:-4}
export TEACHER_POOL=0

# ⚠️ 显存。基线 0.9 是**没有额外常驻模型时**调的：0.9 × 95.5G ≈ 86G 给 vLLM，
# 加上 actor 的 FSDP 分片只剩约 5G 余量 —— HANDOFF §5「step 1~2 正常、第 3 步 OOM」
# 就是这个配方，因为 gpu_memory_utilization 是占**总**显存的比例，不会给后来者让路。
#
# in_process 模式下 teacher 前向发生在 vLLM 睡着之后（sleep_replicas() 已经释放了
# 权重和 KV cache），所以两者在时间上错开、不叠峰值。但 teacher 的 4B 权重仍要常驻
# （param_offload 把参数放 CPU，前向时 gather 分片并放激活），而且训练池现在是 32 卡
# 而非 24 卡，每卡的 FSDP 分片更小但 vLLM 实例更多。
#
# 取 0.7 而不是基线 0.9，也比 ExOPD 的 0.8 再低一档：HANDOFF §5 实测 0.9→0.7 时
# 步时 288→297 s/it（约 3%），代价只是 KV cache 变小，而 KV cache 本来就不是瓶颈。
# 用 3% 的步时换掉「第 3 步才 OOM」那类问题，值得。
export ROLLOUT_GPU_MEM_UTIL=${ROLLOUT_GPU_MEM_UTIL:-0.7}

# ================== in_process 专属门禁 ==================
# 基线脚本自己那几道（Ray 卡数 / 教师目录 / 逐节点 sandbox）继续生效。
# 这里补 mode 这一条，理由和 ExOPD 那边一样：本项目被「命令行里的值 ≠ verl 真正
# 用的值」坑过两次（results.md §6.7 教师 tp 静默变拓扑、§6.3 的 ${VAR:-默认}
# 覆盖时机），所以判据一律是**在解析后的 config 上核对**。
#
# 这一道尤其值得查：teacher 建不出来的话，teacher_logprobs 不进 batch，
# 要等 8 分钟 vLLM 起完才在 loss 里炸。
echo "[check] distillation.mode 与 in-process teacher ..."
# 和基线启动器同一个默认值（那边是 ${TEACHER_MODEL:-...}）。在这里解析掉，
# 免得门禁脚本里再复制一遍 fallback 逻辑。
TEACHER_MODEL=${TEACHER_MODEL:-${B}/outputs_llm_fusion/merge_4b/merged_models/task_arithmetic_scaling_coefficient_1.0}
export TEACHER_MODEL
/opt/conda/envs/torch-base/bin/python - "${TEACHER_MODEL}" <<'PYEOF' || exit 1
import os
import sys

from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf

from verl.trainer.ppo.utils import need_in_process_teacher, need_teacher_pool
from verl.utils import omega_conf_to_dataclass
from verl.workers.config import DistillationConfig, DistillationTeacherModelConfig

teacher_model = sys.argv[1]

cfg_dir = os.path.abspath("verl/trainer/config")
with initialize_config_dir(config_dir=cfg_dir, version_base=None):
    cfg = compose(
        config_name="ppo_trainer",
        overrides=[
            "distillation.enabled=True",
            "distillation.mode=in_process",
            "distillation.n_gpus_per_node=0",
            "distillation.nnodes=0",
            f"distillation.teacher_models.teacher_model.model_path={teacher_model}",
            "distillation.distillation_loss.loss_mode=k1",
            "distillation.distillation_loss.use_policy_gradient=True",
            "distillation.distillation_loss.use_task_rewards=False",
            "actor_rollout_ref.actor.use_kl_loss=False",
            "algorithm.use_kl_in_reward=False",
            # 教师的 inference 子树里 temperature 插值到学生的 rollout.temperature，
            # 而那个键默认是 null。训练脚本传的是 1（on-policy 要求），这里照抄，
            # 否则解析会在无关的插值上失败而不是在我们想检的门禁上。
            "actor_rollout_ref.rollout.temperature=1",
        ],
    )

# 1. 走一遍 DistillationConfig.__post_init__，也就是所有配置期门禁
#    （多教师 / 请求教师池 / top-k loss / 空 model_path / 非法 mode 都在那里 raise）。
#
#    两处刻意的取舍：
#    - 直接构造 dataclass，不用 omega_conf_to_dataclass(cfg.distillation)：后者会把嵌套好的
#      loss_settings dataclass 写回 OmegaConf，撞上 "Cannot assign DistillationLossSettings
#      to Dict[Any, Any]"。
#    - teacher 只构造 model_path 这一个字段，不转 inference 子树：那棵子树里
#      temperature 插值到 ${oc.select:actor_rollout_ref.rollout.temperature}，一旦从
#      根配置上摘下来就失去了解析该插值所需的上下文。反正 in_process 模式下不建任何
#      推理引擎，inference.* 全是惰性的 —— 这也正是本门禁要确认的事实之一。
#    传入值仍然取自解析后的 cfg，所以校验的是 verl 真正会用的那份配置。
d = cfg.distillation
try:
    parsed = DistillationConfig(
        enabled=d.enabled,
        mode=d.mode,
        n_gpus_per_node=d.n_gpus_per_node,
        nnodes=d.nnodes,
        teacher_key=d.teacher_key,
        teacher_models={
            name: DistillationTeacherModelConfig(model_path=sub.model_path)
            for name, sub in d.teacher_models.items()
        },
        distillation_loss=omega_conf_to_dataclass(d.distillation_loss),
    )
except Exception as exc:  # 配置期门禁拒绝
    sys.exit(f"ERROR: DistillationConfig 校验失败：{exc}")

if parsed.mode != "in_process":
    sys.exit(f"ERROR: 解析后的 mode 是 {parsed.mode!r}，不是 'in_process'。")

# 2. teacher 必须真的会在 worker 里被建出来
if not need_in_process_teacher(cfg):
    sys.exit(
        "ERROR: need_in_process_teacher() 是 False —— HF teacher 不会被建起来，\n"
        "       teacher_logprobs 不会进 batch，蒸馏损失无从计算。"
    )

# 3. 教师池必须**不**被创建。建了就是白占 8 张卡，而且不会报错。
if need_teacher_pool(cfg):
    sys.exit("ERROR: need_teacher_pool() 是 True —— 会白建一个教师池占住 8 张卡。")

# 4. teacher 权重目录必须存在（基线脚本也查，但它查的是 vLLM 那条路的默认值；
#    这里 model_path 是我们自己传进去的，早查一步免得 8 分钟后才炸）
teacher = next(iter(parsed.teacher_models.values()))
if not os.path.isfile(os.path.join(teacher.model_path, "config.json")):
    sys.exit(f"ERROR: teacher 权重不是 HF 模型目录（没有 config.json）：{teacher.model_path}")

print(f"  mode         : {parsed.mode}   (解析后)")
print("  teacher 池   : 不创建（need_teacher_pool=False）")
print(f"  teacher 权重 : {teacher.model_path}")
print(f"  loss_mode    : {parsed.distillation_loss.loss_mode}")
print(f"  λ            : {parsed.distillation_loss.reward_extrapolation_lambda}  (1.0 = 不做外推)")
PYEOF

# ================== 交给基线启动器 ==================
export EXPERIMENT_NAME=${EXPERIMENT_NAME:-mopd_hf_teacher_qwen3_4b_${NNODES_OVERRIDE}x8}
export LOG=${LOG:-${B}/logs/llm_fusion/train_mopd_hf_teacher_$(date +%Y%m%d_%H%M%S).log}

cat <<EOF

======== 方案 B：in-process HF teacher ========
  teacher      : 训练进程内冻结 HF 模型（不建 vLLM 教师池）
  拓扑         : ${NNODES_OVERRIDE} 机 × 8 卡 = $(( NNODES_OVERRIDE * 8 )) 全做训练（基线 3×8 训练 + 8 教师）
  teacher 前向 : 每步一次，位置在 ref_log_prob 之后 / update_actor 之前（vLLM 已睡）
  λ            : 1.0（本脚本不做外推；先验证等价性，再叠 ExOPD）
  对照作业     : run_mopd_merged_teacher.sh（vLLM 教师，学生侧其余参数一致）
  实验名       : ${EXPERIMENT_NAME}
  日志         : ${LOG}
  vLLM 显存    : ${ROLLOUT_GPU_MEM_UTIL}（基线 0.9；见脚本注释）

  ⚠️ 首跑要看三件事：
     1. teacher worker 能否构建（CPU 测试覆盖不到这条路径）
     2. teacher logprob 与 vLLM 模式的数值差（不同 kernel，别期望 bitwise）
     3. 步时 vs 基线 —— 决定这条路值不值得的关键数字
===============================================

EOF

exec bash "${REPO}/run_mopd_merged_teacher.sh" "$@"
