#!/bin/bash

# Shared implementation for the fixed CNN/DailyMail source-mask ablations.
set -Eeo pipefail

case "${1:-}:${2:-}" in
  "0.2:mask20" | "0.8:mask80") ;;
  *)
    echo "ERROR: this runner accepts only the fixed mask20 or mask80 configuration." >&2
    exit 2
    ;;
esac
readonly SOURCE_MASK_PROBABILITY="$1"
readonly ABLATION_TAG="$2"
readonly NUM_EPOCHS=4
readonly MAX_STEPS=""

if [[ -z "${SLURM_JOB_ID:-}" ]]; then
  echo "ERROR: submit a mask20 or mask80 launcher with sbatch." >&2
  exit 2
fi

MINIFORGE_MODULE="${TIAO_MINIFORGE_MODULE:-miniforge3/24.1}"
CMAKE_MODULE="${TIAO_CMAKE_MODULE:-cmake/3.31.6}"
MPI_MODULE="${TIAO_MPI_MODULE:-mpi/openmpi4.1.5-gcc11.3.0-cuda11.8-ucx1.12.1}"
CUDNN_MODULE="${TIAO_CUDNN_MODULE:-cudnn/8.6.0.163_cuda11.x}"
CONDA_ENV="${TIAO_CONDA_ENV:-tiao}"

module purge
module load "${MINIFORGE_MODULE}" "${CMAKE_MODULE}"
module load "${MPI_MODULE}" "${CUDNN_MODULE}"
source activate "${CONDA_ENV}"
set -u

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
DEFAULT_REPOSITORY_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
REPOSITORY_ROOT="${TIAO_REPOSITORY_ROOT:-${DEFAULT_REPOSITORY_ROOT}}"
REPOSITORY_ROOT="$(cd "${REPOSITORY_ROOT}" && pwd)"

BASE_MODEL_PATH="${TIAO_BASE_MODEL_PATH:-${REPOSITORY_ROOT}/models/Qwen2.5-7B-Instruct}"
UNIEVAL_MODEL_PATH="${TIAO_UNIEVAL_MODEL_PATH:-${REPOSITORY_ROOT}/models/unieval-sum}"
DATASET_PATH="${TIAO_DATASET_PATH:-${REPOSITORY_ROOT}/data/cnn_dailymail/3.0.0}"
OUTPUT_ROOT="${TIAO_OUTPUT_ROOT:-${REPOSITORY_ROOT}/outputs}"
CACHE_ROOT="${TIAO_CACHE_ROOT:-${REPOSITORY_ROOT}/.cache}"
RUN_ID="${SLURM_JOB_ID}"
MODEL_NAME="${TIAO_MODEL_NAME:-Qwen2.5-7B-Instruct}"
RUN_OUTPUT_DIR="${TIAO_RUN_OUTPUT_DIR:-${OUTPUT_ROOT}/tiao-${ABLATION_TAG}/${RUN_ID}-${MODEL_NAME}}"
FINAL_MODEL_DIR="${TIAO_FINAL_MODEL_DIR:-${RUN_OUTPUT_DIR}/final-model}"
JOB_LOG="${TIAO_JOB_LOG:-${OUTPUT_ROOT}/tiao-${ABLATION_TAG}-training-${RUN_ID}.log}"
RANK_LOG_DIR="${TIAO_RANK_LOG_DIR:-${RUN_OUTPUT_DIR}/torchrun-logs}"
TRITON_CACHE_ROOT="${CACHE_ROOT}/triton"
TORCH_EXTENSIONS_ROOT="${CACHE_ROOT}/torch-extensions"

mkdir -p \
  "${RUN_OUTPUT_DIR}" \
  "${RANK_LOG_DIR}" \
  "${CACHE_ROOT}/huggingface" \
  "${TRITON_CACHE_ROOT}" \
  "${TORCH_EXTENSIONS_ROOT}"
exec > >(tee -a "${JOB_LOG}") 2>&1

on_exit() {
  status=$?
  if [[ ${status} -eq 0 ]]; then
    echo "[$(date --iso-8601=seconds)] TIAO ${ABLATION_TAG} training completed successfully."
    echo "Final model: ${FINAL_MODEL_DIR}"
  else
    echo "[$(date --iso-8601=seconds)] TIAO ${ABLATION_TAG} training failed with exit code ${status}." >&2
    echo "Rank logs: ${RANK_LOG_DIR}" >&2
  fi
}
trap on_exit EXIT

export PYTHONUNBUFFERED=1
unset PYTHONPATH
export NLTK_DATA="${TIAO_NLTK_DATA:-${REPOSITORY_ROOT}/nltk_data}"
export HF_HOME="${HF_HOME:-${CACHE_ROOT}/huggingface}"
export HF_DATASETS_CACHE="${TIAO_DATASETS_CACHE:-${HF_HOME}/datasets/${RUN_ID}}"
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"
export TRANSFORMERS_OFFLINE="${TRANSFORMERS_OFFLINE:-1}"
export HF_DATASETS_OFFLINE="${HF_DATASETS_OFFLINE:-1}"
export TOKENIZERS_PARALLELISM="${TOKENIZERS_PARALLELISM:-false}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
export TORCH_NCCL_ASYNC_ERROR_HANDLING="${TORCH_NCCL_ASYNC_ERROR_HANDLING:-1}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-8}"
mkdir -p "${HF_DATASETS_CACHE}"

# Preload the environment's OpenMP runtime when a binary wheel provides one.
SKLEARN_LIBGOMP_PATH=""
if [[ -n "${CONDA_PREFIX:-}" ]]; then
  for candidate in "${CONDA_PREFIX}"/lib/python*/site-packages/scikit_learn.libs/libgomp-*.so*; do
    if [[ -r "${candidate}" ]]; then
      SKLEARN_LIBGOMP_PATH="${candidate}"
      break
    fi
  done
fi
GCC_LIBGOMP_PATH="$(gcc --print-file-name=libgomp.so.1)"
OPENMP_PRELOAD="${SKLEARN_LIBGOMP_PATH}"
if [[ "${GCC_LIBGOMP_PATH}" != "libgomp.so.1" && -r "${GCC_LIBGOMP_PATH}" ]]; then
  OPENMP_PRELOAD="${OPENMP_PRELOAD:+${OPENMP_PRELOAD}:}${GCC_LIBGOMP_PATH}"
fi
if [[ -n "${OPENMP_PRELOAD}" ]]; then
  export LD_PRELOAD="${OPENMP_PRELOAD}${LD_PRELOAD:+:${LD_PRELOAD}}"
fi

required_files=(
  "${REPOSITORY_ROOT}/tiao_mask_ablation.py"
  "${REPOSITORY_ROOT}/tiao_mask_ablation_trainer.py"
  "${REPOSITORY_ROOT}/tiao.py"
  "${REPOSITORY_ROOT}/tiao_trainer.py"
  "${REPOSITORY_ROOT}/tiao_rollout_trainer.py"
  "${REPOSITORY_ROOT}/unieval.py"
  "${REPOSITORY_ROOT}/utils.py"
  "${REPOSITORY_ROOT}/configs/deepspeed_zero3_offload.json"
  "${BASE_MODEL_PATH}/config.json"
  "${UNIEVAL_MODEL_PATH}/config.json"
  "${UNIEVAL_MODEL_PATH}/model.safetensors"
)
for required_file in "${required_files[@]}"; do
  if [[ ! -f "${required_file}" ]]; then
    echo "ERROR: required file is missing: ${required_file}" >&2
    exit 2
  fi
done
if ! compgen -G "${DATASET_PATH}/train-*.parquet" >/dev/null; then
  echo "ERROR: no CNN/DailyMail training shards were found under ${DATASET_PATH}." >&2
  exit 2
fi
if ! compgen -G "${DATASET_PATH}/validation-*.parquet" >/dev/null; then
  echo "ERROR: no CNN/DailyMail validation shards were found under ${DATASET_PATH}." >&2
  exit 2
fi

mapfile -t NODE_HOSTS < <(scontrol show hostnames "${SLURM_JOB_NODELIST}")
NNODES="${SLURM_NNODES:-8}"
NPROC_PER_NODE="${TIAO_GPUS_PER_NODE:-4}"
WORLD_SIZE=$((NNODES * NPROC_PER_NODE))
MASTER_ADDR="${MASTER_ADDR:-${NODE_HOSTS[0]}}"
MASTER_PORT="${MASTER_PORT:-$((20000 + SLURM_JOB_ID % 20000))}"

# N32-H RoCE/NCCL defaults. Every value can be overridden at submission time.
export NCCL_ALGO="${TIAO_NCCL_ALGO:-${NCCL_ALGO:-Ring}}"
export NCCL_MAX_NCHANNELS="${TIAO_NCCL_MAX_NCHANNELS:-${NCCL_MAX_NCHANNELS:-16}}"
export NCCL_MIN_NCHANNELS="${TIAO_NCCL_MIN_NCHANNELS:-${NCCL_MIN_NCHANNELS:-16}}"
export NCCL_DEBUG="${TIAO_NCCL_DEBUG:-${NCCL_DEBUG:-INFO}}"
export NCCL_IB_HCA="${TIAO_NCCL_IB_HCA:-${NCCL_IB_HCA:-mlx5_0,mlx5_2}}"
export NCCL_IB_GID_INDEX="${TIAO_NCCL_IB_GID_INDEX:-${NCCL_IB_GID_INDEX:-3}}"
export NCCL_IB_TIMEOUT="${TIAO_NCCL_IB_TIMEOUT:-${NCCL_IB_TIMEOUT:-23}}"
export NCCL_IB_RETRY_CNT="${TIAO_NCCL_IB_RETRY_CNT:-${NCCL_IB_RETRY_CNT:-7}}"
if [[ -n "${TIAO_NCCL_TOPO_FILE:-}" ]]; then
  if [[ ! -r "${TIAO_NCCL_TOPO_FILE}" ]]; then
    echo "ERROR: TIAO_NCCL_TOPO_FILE is not readable: ${TIAO_NCCL_TOPO_FILE}" >&2
    exit 2
  fi
  export NCCL_TOPO_FILE="${TIAO_NCCL_TOPO_FILE}"
fi

TRAIN_SAMPLES="${TIAO_TRAIN_SAMPLES:-10000}"
EVAL_SAMPLES="${TIAO_EVAL_SAMPLES:-500}"
LOGGING_STEPS="${TIAO_LOGGING_STEPS:-10}"
SAVE_STEPS="${TIAO_SAVE_STEPS:-100}"
EVAL_STEPS="${TIAO_EVAL_STEPS:-100}"
EVAL_STRATEGY="${TIAO_EVAL_STRATEGY:-steps}"
LEARNING_RATE="${TIAO_LEARNING_RATE:-5e-7}"
MAX_GRAD_NORM="${TIAO_MAX_GRAD_NORM:-0.4}"
MAX_PROMPT_LENGTH="${TIAO_MAX_PROMPT_LENGTH:-2048}"
MAX_COMPLETION_LENGTH="${TIAO_MAX_COMPLETION_LENGTH:-512}"
UNIEVAL_MAX_LENGTH="${TIAO_UNIEVAL_MAX_LENGTH:-1024}"
NUM_GENERATIONS="${TIAO_NUM_GENERATIONS:-8}"
MICRO_BATCH_SIZE="${TIAO_MICRO_BATCH_SIZE:-4}"
GRADIENT_ACCUMULATION_STEPS="${TIAO_GRADIENT_ACCUMULATION_STEPS:-2}"
STEPS_PER_GENERATION="${TIAO_STEPS_PER_GENERATION:-2}"
SCALE_REWARDS="${TIAO_SCALE_REWARDS:-true}"
RESUME_FROM_CHECKPOINT="${TIAO_RESUME_FROM_CHECKPOINT:-}"
SHARED_FS_WAIT_SECONDS="${TIAO_SHARED_FS_WAIT_SECONDS:-120}"
SHARED_FS_POLL_SECONDS="${TIAO_SHARED_FS_POLL_SECONDS:-5}"

for positive_integer_name in \
  NPROC_PER_NODE TRAIN_SAMPLES EVAL_SAMPLES NUM_EPOCHS LOGGING_STEPS SAVE_STEPS \
  EVAL_STEPS MAX_PROMPT_LENGTH MAX_COMPLETION_LENGTH UNIEVAL_MAX_LENGTH \
  NUM_GENERATIONS MICRO_BATCH_SIZE GRADIENT_ACCUMULATION_STEPS \
  STEPS_PER_GENERATION SHARED_FS_WAIT_SECONDS SHARED_FS_POLL_SECONDS; do
  if [[ ! "${!positive_integer_name}" =~ ^[1-9][0-9]*$ ]]; then
    echo "ERROR: ${positive_integer_name} must be a positive integer." >&2
    exit 2
  fi
done
if [[ -n "${MAX_STEPS}" && ! "${MAX_STEPS}" =~ ^[1-9][0-9]*$ ]]; then
  echo "ERROR: TIAO_MAX_STEPS must be empty or a positive integer." >&2
  exit 2
fi
if [[ "${EVAL_STRATEGY}" != "steps" && "${EVAL_STRATEGY}" != "epoch" && "${EVAL_STRATEGY}" != "no" ]]; then
  echo "ERROR: TIAO_EVAL_STRATEGY must be steps, epoch, or no." >&2
  exit 2
fi
if [[ "${SCALE_REWARDS}" != "true" && "${SCALE_REWARDS}" != "false" ]]; then
  echo "ERROR: TIAO_SCALE_REWARDS must be true or false." >&2
  exit 2
fi
if [[ -n "${RESUME_FROM_CHECKPOINT}" && ! -f "${RESUME_FROM_CHECKPOINT}/trainer_state.json" ]]; then
  echo "ERROR: the resume checkpoint is missing trainer_state.json: ${RESUME_FROM_CHECKPOINT}" >&2
  exit 2
fi

EFFECTIVE_TRAIN_BATCH=$((WORLD_SIZE * MICRO_BATCH_SIZE * GRADIENT_ACCUMULATION_STEPS))
GENERATION_BATCH=$((WORLD_SIZE * MICRO_BATCH_SIZE * STEPS_PER_GENERATION))
if (( EFFECTIVE_TRAIN_BATCH % NUM_GENERATIONS != 0 || GENERATION_BATCH % NUM_GENERATIONS != 0 )); then
  echo "ERROR: TIAO_NUM_GENERATIONS must divide both effective batch sizes." >&2
  exit 2
fi

export REPOSITORY_ROOT BASE_MODEL_PATH UNIEVAL_MODEL_PATH DATASET_PATH
export RUN_ID RUN_OUTPUT_DIR FINAL_MODEL_DIR RANK_LOG_DIR
export TRITON_CACHE_ROOT TORCH_EXTENSIONS_ROOT
export NNODES NPROC_PER_NODE WORLD_SIZE MASTER_ADDR MASTER_PORT
export TRAIN_SAMPLES EVAL_SAMPLES NUM_EPOCHS LOGGING_STEPS SAVE_STEPS EVAL_STEPS
export EVAL_STRATEGY MAX_STEPS LEARNING_RATE MAX_GRAD_NORM MAX_PROMPT_LENGTH
export MAX_COMPLETION_LENGTH UNIEVAL_MAX_LENGTH NUM_GENERATIONS MICRO_BATCH_SIZE
export GRADIENT_ACCUMULATION_STEPS STEPS_PER_GENERATION SCALE_REWARDS
export RESUME_FROM_CHECKPOINT SHARED_FS_WAIT_SECONDS SHARED_FS_POLL_SECONDS
export SOURCE_MASK_PROBABILITY ABLATION_TAG

echo "============================================================"
echo "TIAO source-mask ablation on CNN/DailyMail (${ABLATION_TAG})"
echo "world size:              ${WORLD_SIZE} (${NNODES} nodes x ${NPROC_PER_NODE} GPUs)"
echo "base model:              ${BASE_MODEL_PATH}"
echo "training samples:        ${TRAIN_SAMPLES}"
echo "validation samples:      ${EVAL_SAMPLES}"
echo "source mask probability: ${SOURCE_MASK_PROBABILITY}"
echo "epochs:                  ${NUM_EPOCHS}"
echo "effective train batch:   ${EFFECTIVE_TRAIN_BATCH}"
echo "generation batch:        ${GENERATION_BATCH}"
echo "micro batch per rank:    ${MICRO_BATCH_SIZE}"
echo "gradient accumulation:   ${GRADIENT_ACCUMULATION_STEPS}"
echo "steps per generation:    ${STEPS_PER_GENERATION}"
echo "generations per prompt:  ${NUM_GENERATIONS}"
echo "learning rate:           ${LEARNING_RATE}"
echo "maximum gradient norm:   ${MAX_GRAD_NORM}"
echo "reference KL beta:       0"
echo "output:                  ${RUN_OUTPUT_DIR}"
echo "============================================================"

srun \
  --nodes="${NNODES}" \
  --ntasks="${NNODES}" \
  --ntasks-per-node=1 \
  --gres="gpu:${NPROC_PER_NODE}" \
  --chdir=/tmp \
  --kill-on-bad-exit=1 \
  /bin/bash -c '
    set -Eeo pipefail
    NODE_RANK="${SLURM_PROCID}"
    NODE_NAME="$(hostname)"
    deadline=$((SECONDS + SHARED_FS_WAIT_SECONDS))
    while [[ ! -r "${REPOSITORY_ROOT}/tiao_mask_ablation.py" \
         || ! -r "${REPOSITORY_ROOT}/tiao_mask_ablation_trainer.py" \
         || ! -r "${BASE_MODEL_PATH}/config.json" \
         || ! -r "${UNIEVAL_MODEL_PATH}/model.safetensors" ]]; do
      if (( SECONDS >= deadline )); then
        echo "ERROR: required shared files remain unreadable on ${NODE_NAME}." >&2
        exit 72
      fi
      sleep "${SHARED_FS_POLL_SECONDS}"
    done

    NODE_RANK_LOG_DIR="${RANK_LOG_DIR}/node-${NODE_RANK}-${NODE_NAME}"
    export TRITON_CACHE_DIR="${TRITON_CACHE_ROOT}/${RUN_ID}/node-${NODE_RANK}"
    export TORCH_EXTENSIONS_DIR="${TORCH_EXTENSIONS_ROOT}/${NODE_NAME}"
    mkdir -p "${NODE_RANK_LOG_DIR}" "${TRITON_CACHE_DIR}" "${TORCH_EXTENSIONS_DIR}"
    cd "${REPOSITORY_ROOT}"

    extra_args=(
      --dataset_sample_train_num "${TRAIN_SAMPLES}"
      --dataset_sample_eval_num "${EVAL_SAMPLES}"
    )
    if [[ -n "${RESUME_FROM_CHECKPOINT}" ]]; then
      extra_args+=(--resume_from_checkpoint "${RESUME_FROM_CHECKPOINT}")
    fi
    if [[ -n "${MAX_STEPS}" ]]; then
      extra_args+=(--max_steps "${MAX_STEPS}")
    fi
    if [[ "${NODE_RANK}" -eq 0 ]]; then
      log_args=(--tee=3 --local-ranks-filter=0)
    else
      log_args=(--redirects=3)
    fi

    torchrun \
      --nnodes="${NNODES}" \
      --nproc_per_node="${NPROC_PER_NODE}" \
      --node_rank="${NODE_RANK}" \
      --master_addr="${MASTER_ADDR}" \
      --master_port="${MASTER_PORT}" \
      --log-dir="${NODE_RANK_LOG_DIR}" \
      "${log_args[@]}" \
      tiao_mask_ablation.py \
      --output_dir "${RUN_OUTPUT_DIR}" \
      --final_model_output_dir "${FINAL_MODEL_DIR}" \
      --overwrite_output_dir false \
      --learning_rate "${LEARNING_RATE}" \
      --adam_beta1 0.9 \
      --adam_beta2 0.999 \
      --weight_decay 0.1 \
      --warmup_ratio 0.1 \
      --lr_scheduler_type cosine \
      --logging_steps "${LOGGING_STEPS}" \
      --bf16 true \
      --tf32 true \
      --per_device_eval_batch_size 1 \
      --per_device_train_batch_size "${MICRO_BATCH_SIZE}" \
      --gradient_accumulation_steps "${GRADIENT_ACCUMULATION_STEPS}" \
      --gradient_checkpointing true \
      --num_generations "${NUM_GENERATIONS}" \
      --steps_per_generation "${STEPS_PER_GENERATION}" \
      --num_train_epochs "${NUM_EPOCHS}" \
      --save_steps "${SAVE_STEPS}" \
      --eval_steps "${EVAL_STEPS}" \
      --eval_strategy "${EVAL_STRATEGY}" \
      --max_prompt_length "${MAX_PROMPT_LENGTH}" \
      --max_completion_length "${MAX_COMPLETION_LENGTH}" \
      --max_grad_norm "${MAX_GRAD_NORM}" \
      --deepspeed "${REPOSITORY_ROOT}/configs/deepspeed_zero3_offload.json" \
      --temperature 1 \
      --repetition_penalty 1 \
      --beta 0 \
      --scale_rewards "${SCALE_REWARDS}" \
      --num_completions_to_print 1 \
      --unieval_model_name_or_path "${UNIEVAL_MODEL_PATH}" \
      --unieval_model_deepspeed_config none \
      --unieval_max_length "${UNIEVAL_MAX_LENGTH}" \
      --base_model_name_or_path "${BASE_MODEL_PATH}" \
      --dataset_path "${DATASET_PATH}" \
      --source_mask_probability "${SOURCE_MASK_PROBABILITY}" \
      --random_seed 1999 \
      --log_completions true \
      --report_to none \
      "${extra_args[@]}"
  '
