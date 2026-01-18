#!/bin/bash

# ================= Configuration =================
CONTAINER_NAME="jdiff_work" 

DOCKER_WORK_ROOT="/workspace/"
HOST_WORK_ROOT="$( cd "$( dirname "$( dirname "${BASH_SOURCE[0]}" )" )" &> /dev/null && pwd )"

OUTPUT_DIR="outputs/"

# conda env
DOCKER_RUN_ENV="jdiffusion"
HOST_EVAL_ENV="jdiff_eval"

DOCKER_ENV_CMD="source /root/anaconda3/etc/profile.d/conda.sh && conda activate $DOCKER_RUN_ENV && cd $DOCKER_WORK_ROOT"
# =================================================

function clean_up() {
    echo -e "\n\n[Pipeline] !!! INTERRUPTED !!! Killing processes..."
    docker exec $CONTAINER_NAME pkill -INT -f "python scripts" || true
    pkill -INT -f "python evaluation" || true
    exit 1
}
trap clean_up SIGINT SIGTERM


# Usage: ./auto_pipeline.sh 5
NUM_STYLES=15
if [ ! -z "$1" ]; then
    NUM_STYLES=$1
fi

# stop on errors
set -e 

echo -e "\n=============================================="
echo "Start Auto Pipeline: Training -> Inference -> Evaluation"
echo "Target Styles: $NUM_STYLES"
echo "Container: $CONTAINER_NAME"
echo "=============================================="

cd "$HOST_WORK_ROOT"
echo "[Pipeline] Switch to work space: $HOST_WORK_ROOT"

TOTAL_START_TIME=$(date +%s)
STEP_START_TIME=0

# ------- Helper Functions -------

function timer_start() {
    STEP_START_TIME=$(date +%s)
}

function timer_end() {
    local step_name=$1
    local end_time=$(date +%s)
    local duration=$((end_time - STEP_START_TIME))
    # Convert to minutes and seconds
    local min=$((duration / 60))
    local sec=$((duration % 60))
    
    echo "[Pipeline] ($step_name) Time Elapsed: ${min}m ${sec}s (${duration}s)"
}

function activate_host_env() {
    if [[ "$CONDA_DEFAULT_ENV" != "$HOST_EVAL_ENV" ]]; then
        echo "[Env] Activating host conda env: $HOST_EVAL_ENV..."
        eval "$(conda shell.bash hook)"
        conda activate $HOST_EVAL_ENV
    fi
}

# --- Stage Definitions ---

function stage_train() {
    echo -e "\n[1/5] Start Training in Docker (train_all.py)..."
    timer_start
    docker exec $CONTAINER_NAME /bin/bash -c "$DOCKER_ENV_CMD && python scripts/train_all.py -n $NUM_STYLES"
    # docker exec $CONTAINER_NAME /bin/bash -c "$DOCKER_ENV_CMD && mkdir -p test_dir"
    timer_end "Training Stage"
}

function stage_infer() {
    echo -e "\n[2/5] Start Inference in Docker (infer_all.py)..."
    timer_start
    docker exec $CONTAINER_NAME /bin/bash -c "$DOCKER_ENV_CMD && python scripts/infer_all.py -n $NUM_STYLES"
    # docker exec $CONTAINER_NAME /bin/bash -c "$DOCKER_ENV_CMD && ls -l test_dir"
    timer_end "Inference Stage"
}

function stage_permission() {
    echo -e "\n[3/5] Fixing permissions for outputs directory..."
    local uid=$(id -u)
    local gid=$(id -g)
    docker exec $CONTAINER_NAME /bin/bash -c "$DOCKER_ENV_CMD && chown -R $uid:$gid $DOCKER_WORK_ROOT/$OUTPUT_DIR"
    echo "[Pipeline] Permissions fixed."
}

function stage_eval() {
    echo -e "\n[4/5] Running fine-grained evaluation (run_eval.py)..."
    activate_host_env
    timer_start
    python evaluation/run_eval.py -n $NUM_STYLES
    # rmdir $HOST_WORK_ROOT/test_dir
    timer_end "Evaluation Stage"
}

function stage_report() {
    echo -e "\n[5/5] Generating final report (eval_score.py)..."
    activate_host_env
    python evaluation/eval_score.py
    # ls -l
}

# =================================================
#                 Main Execution
#       Comment out lines to skip stages
# =================================================

stage_train
stage_infer
stage_permission
stage_eval
stage_report

# --- Final: Time Consumption ---
TOTAL_END_TIME=$(date +%s)
TOTAL_DURATION=$((TOTAL_END_TIME - TOTAL_START_TIME))
TOTAL_MIN=$((TOTAL_DURATION / 60))
TOTAL_SEC=$((TOTAL_DURATION % 60))

echo -e "\n=============================================="
echo "[Pipeline] All Done!"
echo "[Pipeline] Total Time: ${TOTAL_MIN}m ${TOTAL_SEC}s"
echo "=============================================="