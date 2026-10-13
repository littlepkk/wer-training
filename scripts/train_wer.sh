#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/.." && pwd)"
export VERL_REPO="${REPO_ROOT}/verl"
export PYTHONPATH="${VERL_REPO}:${VERL_REPO}/berkeley-function-call-leaderboard:${VERL_REPO}/tau2-bench-verified/src${PYTHONPATH:+:${PYTHONPATH}}"

export BASE_MODEL="${BASE_MODEL:-/path/to/models/skill-optimizer}"
export LOCAL_MODEL_PATH="${LOCAL_MODEL_PATH:-${BASE_MODEL}}"
export TOKENIZER_MODEL_PATH="${TOKENIZER_MODEL_PATH:-${BASE_MODEL}}"
export WER_DATA_DIR="${WER_DATA_DIR:-${REPO_ROOT}/extra_datasets}"
TRAIN_FILE="${TRAIN_FILE:-${WER_DATA_DIR}/origin/train.parquet}"
export PROJECT_NAME="${PROJECT_NAME:-wer_training}"
export EXPERIMENT_NAME="${EXPERIMENT_NAME:-wer_run}"
OUTPUT_DIR="${OUTPUT_DIR:-${REPO_ROOT}/outputs/${EXPERIMENT_NAME}}"
TRAIN_DEVICE="${TRAIN_DEVICE:-npu}"
N_GPUS_PER_NODE="${N_GPUS_PER_NODE:-6}"
NNODES="${NNODES:-1}"

export VLLM_USE_V1="${VLLM_USE_V1:-1}"
export BFCL_PROJECT_ROOT="${BFCL_PROJECT_ROOT:-${OUTPUT_DIR}/reward}"
export TAU2_DATA_DIR="${TAU2_DATA_DIR:-${VERL_REPO}/tau2-bench-verified/data}"
export WER_EXECUTOR_BASE_URL="${WER_EXECUTOR_BASE_URL:-http://localhost:8000/v1}"
export WER_EXECUTOR_API_KEY="${WER_EXECUTOR_API_KEY:-EMPTY}"
export WER_EXECUTOR_MODEL="${WER_EXECUTOR_MODEL:-executor-model}"

if [[ "${TRAIN_DEVICE}" == "npu" ]]; then
    export ASCEND_RT_VISIBLE_DEVICES="${ASCEND_RT_VISIBLE_DEVICES:-0,1,2,3,4,5}"
    export RAY_EXPERIMENTAL_NOSET_ASCEND_RT_VISIBLE_DEVICES=1
    export VLLM_ATTENTION_BACKEND="${VLLM_ATTENTION_BACKEND:-XFORMERS}"
fi

mkdir -p "${OUTPUT_DIR}/logs" "${OUTPUT_DIR}/rollouts" "${OUTPUT_DIR}/checkpoints"

python3 -m verl.trainer.main_ppo \
    --config-path "${VERL_REPO}/verl/trainer/config" \
    --config-name ppo_trainer \
    algorithm.adv_estimator=grpo \
    algorithm.norm_adv_by_std_in_grpo=False \
    "data.train_files='${TRAIN_FILE}'" \
    "data.val_files='${TRAIN_FILE}'" \
    data.train_batch_size=6 \
    data.return_raw_chat=True \
    data.max_prompt_length=19000 \
    data.max_response_length=4096 \
    "actor_rollout_ref.model.path='${BASE_MODEL}'" \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.actor.optim.lr=1e-6 \
    actor_rollout_ref.actor.optim.warmup_style=cosine \
    actor_rollout_ref.actor.ppo_mini_batch_size=6 \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=1 \
    actor_rollout_ref.actor.ulysses_sequence_parallel_size=2 \
    actor_rollout_ref.actor.use_kl_loss=False \
    actor_rollout_ref.actor.kl_loss_coef=0.0 \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.actor.fsdp_config.param_offload=False \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=False \
    "actor_rollout_ref.actor.checkpoint.save_contents=['hf_model']" \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=1 \
    actor_rollout_ref.rollout.tensor_model_parallel_size=2 \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.top_k=50 \
    actor_rollout_ref.rollout.temperature=0.9 \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.7 \
    actor_rollout_ref.rollout.n=4 \
    actor_rollout_ref.rollout.mode=async \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=1 \
    actor_rollout_ref.ref.fsdp_config.param_offload=True \
    "custom_reward_function.path='${VERL_REPO}/berkeley-function-call-leaderboard/bfcl_eval/skill.py'" \
    custom_reward_function.name=compute_score \
    algorithm.kl_ctrl.kl_coef=0.0 \
    "trainer.rollout_data_dir='${OUTPUT_DIR}/rollouts'" \
    "trainer.logger=['console']" \
    "trainer.project_name='${PROJECT_NAME}'" \
    "trainer.experiment_name='${EXPERIMENT_NAME}'" \
    "trainer.device=${TRAIN_DEVICE}" \
    "trainer.n_gpus_per_node=${N_GPUS_PER_NODE}" \
    "trainer.nnodes=${NNODES}" \
    trainer.phase_training.enable=True \
    trainer.phase_training.num_phases=3 \
    'trainer.phase_training.epochs=[2,2,2]' \
    "trainer.phase_training.saved_data_path_template='${WER_DATA_DIR}/{phase_id}/failed_skill_records.jsonl'" \
    "trainer.phase_training.generated_train_files_template='${WER_DATA_DIR}/{phase_id}/train.parquet'" \
    trainer.save_freq=8 \
    trainer.test_freq=-1 \
    trainer.total_epochs=2 \
    trainer.val_before_train=False \
    trainer.log_val_generations=0 \
    "trainer.default_local_dir='${OUTPUT_DIR}/checkpoints'" \
    "$@" \
    2>&1 | tee "${OUTPUT_DIR}/logs/train.log"
