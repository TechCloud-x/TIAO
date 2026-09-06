#!/bin/bash
#SBATCH --job-name=TIAO-CD-inference
#SBATCH --nodes=8
#SBATCH --ntasks=8
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:4
#SBATCH --exclude=paraai-n32-h-01-agent-92
#SBATCH --cpus-per-task=128
#SBATCH --qos=gpugpu
#SBATCH --output=slurm-%j.out

# Distributed CNN/DailyMail test inference: 8 nodes x 4 GPUs = 32 ranks.
set -Eeo pipefail

if [[ -z "${SLURM_JOB_ID:-}" ]]; then
  echo "ERROR: submit this launcher with sbatch." >&2
  exit 2
fi
if [[ -z "${INFERENCE_MODEL_PATH:-}" ]]; then
  echo "ERROR: INFERENCE_MODEL_PATH must identify a trained checkpoint or final model." >&2
  echo "Usage: sbatch --export=ALL,INFERENCE_MODEL_PATH=/path/to/model $0" >&2
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
MODEL_PATH="${INFERENCE_MODEL_PATH%/}"
MODEL_DIRECTORY_NAME="${MODEL_PATH##*/}"
MODEL_PARENT_PATH="${MODEL_PATH%/*}"
MODEL_PARENT_NAME="${MODEL_PARENT_PATH##*/}"
MODEL_LABEL="${TIAO_INFERENCE_MODEL_LABEL:-${MODEL_PARENT_NAME}-${MODEL_DIRECTORY_NAME}}"
UNIEVAL_MODEL_PATH="${TIAO_UNIEVAL_MODEL_PATH:-${REPOSITORY_ROOT}/models/unieval-sum}"
DATASET_PATH="${TIAO_DATASET_PATH:-${REPOSITORY_ROOT}/data/cnn_dailymail/3.0.0}"
OUTPUT_ROOT="${TIAO_OUTPUT_ROOT:-${REPOSITORY_ROOT}/outputs}"
CACHE_ROOT="${TIAO_CACHE_ROOT:-${REPOSITORY_ROOT}/.cache}"
RUN_ID="${SLURM_JOB_ID}"
RUN_OUTPUT_DIR="${TIAO_INFERENCE_OUTPUT_DIR:-${OUTPUT_ROOT}/inference/${RUN_ID}-${MODEL_LABEL}}"
JOB_LOG="${TIAO_INFERENCE_JOB_LOG:-${OUTPUT_ROOT}/tiao-inference-${RUN_ID}.log}"
RANK_LOG_DIR="${TIAO_INFERENCE_RANK_LOG_DIR:-${RUN_OUTPUT_DIR}/torchrun-logs}"
TORCH_EXTENSIONS_ROOT="${CACHE_ROOT}/torch-extensions"
TRITON_CACHE_ROOT="${CACHE_ROOT}/triton"

mkdir -p \
  "${RUN_OUTPUT_DIR}" \
  "${RANK_LOG_DIR}" \
  "${CACHE_ROOT}/huggingface" \
  "${TORCH_EXTENSIONS_ROOT}" \
  "${TRITON_CACHE_ROOT}"
exec > >(tee -a "${JOB_LOG}") 2>&1

on_exit() {
  status=$?
  if [[ ${status} -eq 0 ]]; then
    echo "[$(date --iso-8601=seconds)] CNN/DailyMail inference completed successfully."
    echo "Metrics: ${RUN_OUTPUT_DIR}/metrics.json"
    echo "Predictions: ${RUN_OUTPUT_DIR}/predictions.jsonl"
  else
    echo "[$(date --iso-8601=seconds)] CNN/DailyMail inference failed with exit code ${status}." >&2
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

TEST_SAMPLES="${TIAO_INFERENCE_TEST_SAMPLES:-all}"
GENERATION_BATCH_SIZE="${TIAO_INFERENCE_GENERATION_BATCH_SIZE:-4}"
EVALUATION_BATCH_SIZE="${TIAO_INFERENCE_EVALUATION_BATCH_SIZE:-8}"
MAX_PROMPT_LENGTH="${TIAO_INFERENCE_MAX_PROMPT_LENGTH:-2048}"
MAX_NEW_TOKENS="${TIAO_INFERENCE_MAX_NEW_TOKENS:-512}"
UNIEVAL_MAX_LENGTH="${TIAO_UNIEVAL_MAX_LENGTH:-1024}"
RANDOM_SEED="${TIAO_INFERENCE_SEED:-2025}"
EVALUATE_UNIEVAL="${TIAO_INFERENCE_EVALUATE_UNIEVAL:-true}"
SHARED_FS_WAIT_SECONDS="${TIAO_SHARED_FS_WAIT_SECONDS:-120}"
SHARED_FS_POLL_SECONDS="${TIAO_SHARED_FS_POLL_SECONDS:-5}"

if [[ "${TEST_SAMPLES}" != "all" && ! "${TEST_SAMPLES}" =~ ^[1-9][0-9]*$ ]]; then
  echo "ERROR: TIAO_INFERENCE_TEST_SAMPLES must be a positive integer or all." >&2
  exit 2
fi
for positive_integer_name in \
  GENERATION_BATCH_SIZE EVALUATION_BATCH_SIZE MAX_PROMPT_LENGTH MAX_NEW_TOKENS \
  UNIEVAL_MAX_LENGTH SHARED_FS_WAIT_SECONDS SHARED_FS_POLL_SECONDS; do
  if [[ ! "${!positive_integer_name}" =~ ^[1-9][0-9]*$ ]]; then
    echo "ERROR: ${positive_integer_name} must be a positive integer." >&2
    exit 2
  fi
done
if [[ "${EVALUATE_UNIEVAL}" != "true" && "${EVALUATE_UNIEVAL}" != "false" ]]; then
  echo "ERROR: TIAO_INFERENCE_EVALUATE_UNIEVAL must be true or false." >&2
  exit 2
fi

required_files=(
  "${REPOSITORY_ROOT}/inference_cnn_dailymail.py"
  "${REPOSITORY_ROOT}/unieval.py"
  "${REPOSITORY_ROOT}/utils.py"
  "${MODEL_PATH}/config.json"
)
if [[ "${EVALUATE_UNIEVAL}" == "true" ]]; then
  required_files+=("${UNIEVAL_MODEL_PATH}/config.json" "${UNIEVAL_MODEL_PATH}/model.safetensors")
fi
for required_file in "${required_files[@]}"; do
  if [[ ! -f "${required_file}" ]]; then
    echo "ERROR: required file is missing: ${required_file}" >&2
    exit 2
  fi
done
if [[ ! -f "${MODEL_PATH}/model.safetensors" && ! -f "${MODEL_PATH}/model.safetensors.index.json" ]]; then
  echo "ERROR: no safetensors model weights were found under ${MODEL_PATH}." >&2
  exit 2
fi
if ! compgen -G "${DATASET_PATH}/test-*.parquet" >/dev/null; then
  echo "ERROR: no CNN/DailyMail test shards were found under ${DATASET_PATH}." >&2
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

export REPOSITORY_ROOT MODEL_PATH MODEL_LABEL UNIEVAL_MODEL_PATH DATASET_PATH
export RUN_ID RUN_OUTPUT_DIR RANK_LOG_DIR TORCH_EXTENSIONS_ROOT TRITON_CACHE_ROOT
export TEST_SAMPLES GENERATION_BATCH_SIZE EVALUATION_BATCH_SIZE MAX_PROMPT_LENGTH
export MAX_NEW_TOKENS UNIEVAL_MAX_LENGTH RANDOM_SEED EVALUATE_UNIEVAL
export SHARED_FS_WAIT_SECONDS SHARED_FS_POLL_SECONDS
export NNODES NPROC_PER_NODE WORLD_SIZE MASTER_ADDR MASTER_PORT

echo "============================================================"
echo "TIAO checkpoint inference on CNN/DailyMail test"
echo "world size:          ${WORLD_SIZE} (${NNODES} nodes x ${NPROC_PER_NODE} GPUs)"
echo "model:               ${MODEL_PATH}"
echo "test samples:        ${TEST_SAMPLES}"
echo "generation batch:    ${GENERATION_BATCH_SIZE} per rank"
echo "evaluation batch:    ${EVALUATION_BATCH_SIZE} per rank"
echo "global averaging:    sum of sample scores / global sample count"
echo "evaluate UniEval:    ${EVALUATE_UNIEVAL}"
echo "output:              ${RUN_OUTPUT_DIR}"
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
    while [[ ! -r "${REPOSITORY_ROOT}/inference_cnn_dailymail.py" \
         || ! -r "${MODEL_PATH}/config.json" ]]; do
      if (( SECONDS >= deadline )); then
        echo "ERROR: required shared files remain unreadable on ${NODE_NAME}." >&2
        exit 72
      fi
      sleep "${SHARED_FS_POLL_SECONDS}"
    done

    NODE_RANK_LOG_DIR="${RANK_LOG_DIR}/node-${NODE_RANK}-${NODE_NAME}"
    export TORCH_EXTENSIONS_DIR="${TORCH_EXTENSIONS_ROOT}/${NODE_NAME}"
    export TRITON_CACHE_DIR="${TRITON_CACHE_ROOT}/${RUN_ID}/node-${NODE_RANK}"
    mkdir -p "${NODE_RANK_LOG_DIR}" "${TORCH_EXTENSIONS_DIR}" "${TRITON_CACHE_DIR}"
    cd "${REPOSITORY_ROOT}"

    evaluation_args=()
    if [[ "${EVALUATE_UNIEVAL}" == "false" ]]; then
      evaluation_args+=(--skip-unieval)
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
      inference_cnn_dailymail.py \
      --model-path "${MODEL_PATH}" \
      --model-label "${MODEL_LABEL}" \
      --unieval-model-path "${UNIEVAL_MODEL_PATH}" \
      --dataset-path "${DATASET_PATH}" \
      --output-dir "${RUN_OUTPUT_DIR}" \
      --test-samples "${TEST_SAMPLES}" \
      --seed "${RANDOM_SEED}" \
      --generation-batch-size "${GENERATION_BATCH_SIZE}" \
      --evaluation-batch-size "${EVALUATION_BATCH_SIZE}" \
      --max-prompt-length "${MAX_PROMPT_LENGTH}" \
      --max-new-tokens "${MAX_NEW_TOKENS}" \
      --unieval-max-length "${UNIEVAL_MAX_LENGTH}" \
      "${evaluation_args[@]}"
  '
