#!/bin/bash
# Run the full standard evaluation suite sequentially.
# Usage: bash script/eval_all.sh --model qwen2_5_vl --model_args "pretrained=Qwen/Qwen2.5-VL-3B-Instruct,device_map=cuda:0"
#bash script/eval_all.sh --model bagel --model_args "pretrained=ByteDance-Seed/BAGEL-7B-MoT,save_intermediate=true,device_map=cuda:0"
set -e

MODEL=""
MODEL_ARGS=""

while [[ $# -gt 0 ]]; do
    case "$1" in
        --model)      MODEL="$2";      shift 2 ;;
        --model_args) MODEL_ARGS="$2"; shift 2 ;;
        *) echo "Unknown arg: $1"; exit 1 ;;
    esac
done

if [[ -z "$MODEL" ]]; then
    echo "Usage: bash script/eval_all.sh --model <model> --model_args <args>"
    exit 1
fi

OUTPUT_BASE="./logs/${MODEL}"
mkdir -p "$OUTPUT_BASE"

export WORLD_SIZE=1
export RANK=0
export LOCAL_RANK=0
export MASTER_ADDR=127.0.0.1
export MASTER_PORT=29314

TASKS=(
    auxsolidmath_easy
    chartqa100
    geometry3k
    babyvision
    illusionbench_arshia_test
    mmsi_attribute_appr
    mmsi_attribute_meas
    mmsi_motion_cam
    mmsi_motion_obj
    mmsi_msr
    phyx_simple
    realunify
    uni_mmmu
    vsp
    VisualPuzzles
)

for TASK in "${TASKS[@]}"; do
    echo "========================================"
    echo "Running: $TASK"
    echo "========================================"
    uv run python -m lmms_eval \
        --model "$MODEL" \
        --model_args "$MODEL_ARGS" \
        --tasks "$TASK" \
        --batch_size 1 \
        --log_samples \
        --output_path "${OUTPUT_BASE}/${TASK}" 
        
    echo "Done: $TASK"
    echo
done

echo "All tasks completed. Results in ${OUTPUT_BASE}/"

# Aggregate all results.
echo ""
echo "========================================"
echo "SUMMARY"
echo "========================================"
uv run python script/aggregate_results.py --output-base "$OUTPUT_BASE" --mode standard
