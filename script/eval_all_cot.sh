#!/bin/bash
# Run the full CoT / Visual CoT evaluation suite sequentially.
# Usage: bash script/eval_all_cot.sh --model bagel_visual_cot --model_args "pretrained=ByteDance-Seed/BAGEL-7B-MoT,save_intermediate=true,device_map=cuda:0"

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
    echo "Usage: bash script/eval_all_cot.sh --model <model> --model_args <args>"
    exit 1
fi

OUTPUT_BASE="./logs/${MODEL}"
mkdir -p "$OUTPUT_BASE"

# Force single-node, single-process distributed settings to avoid
# accidentally attaching to an external distributed environment.


TASKS=(
    auxsolidmath_easy_visual_cot
    chartqa100_visual_cot
    geometry3k_visual_cot
    babyvision_cot
    illusionbench_arshia_visual_cot_split
    mmsi_attribute_appr_visual_cot
    mmsi_attribute_meas_visual_cot
    mmsi_motion_cam_visual_cot
    mmsi_motion_obj_visual_cot
    mmsi_msr_visual_cot
    phyx_cot
    realunify_cot
    uni_mmmu_cot
    vsp_cot
    VisualPuzzles_visual_cot
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
        --output_path "${OUTPUT_BASE}/${TASK}" \
        --limit 1
    echo "Done: $TASK"
    echo
done

echo "All tasks completed. Results in ${OUTPUT_BASE}/"

# Aggregate all results.
echo ""
echo "========================================"
echo "SUMMARY"
echo "========================================"
uv run python script/aggregate_results.py --output-base "$OUTPUT_BASE" --mode cot
