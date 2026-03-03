# One-shot Evaluation Scripts

The `script/` directory contains two sequential "run the whole suite once"
helpers:

- `eval_all.sh`: standard task suite
- `eval_all_cot.sh`: CoT / Visual CoT task suite

Both scripts take a model already registered in `lmms_eval` plus its
`model_args`, run a fixed task list one by one, and generate final flat,
category, and benchmark summaries after all runs finish.

## Recommended Usage

Run the scripts from the repository root instead of `cd script` first. That
keeps outputs under the root-level `logs/` directory.

Standard suite:

```bash
bash script/eval_all.sh \
  --model qwen2_5_vl \
  --model_args "pretrained=Qwen/Qwen2.5-VL-3B-Instruct"
```

CoT suite:

```bash
bash script/eval_all_cot.sh \
  --model bagel_visual_cot \
  --model_args "pretrained=ByteDance-Seed/BAGEL-7B-MoT,save_intermediate=true"
```

## Parameters

Both scripts accept exactly two arguments:

- `--model`: a model name registered in `lmms_eval`, for example
  `qwen2_5_vl` or `bagel_visual_cot`
- `--model_args`: the model argument string passed through to `lmms_eval`, for
  example `pretrained=...`, `device_map=auto`, or `save_intermediate=true`

Under the hood, each task is executed with:

```bash
uv run python -m lmms_eval \
  --model "$MODEL" \
  --model_args "$MODEL_ARGS" \
  --tasks "$TASK" \
  --batch_size 1 \
  --log_samples \
  --output_path "${OUTPUT_BASE}/${TASK}"
```

## Prerequisites

- The repository dependencies are already installed in the current environment
- `uv` is available from the command line
- The target model is implemented in `lmms_eval` and `--model_args` is valid
- Required datasets, access permissions, and API keys have been configured as
  described in the main project README

If your environment is not managed with `uv`, replace `uv run python` in the
scripts with your preferred Python launcher.

## Script Behavior

Both scripts:

- Run a fixed task list sequentially
- Override distributed environment variables to stay on a local single-node,
  single-process setup
- Always use `--batch_size 1`
- Always enable `--log_samples`
- Exit immediately on the first failed task because they use `set -e`
- Scan result files and generate benchmark summaries after all tasks finish

The scripts overwrite these environment variables:

```bash
export WORLD_SIZE=1
export RANK=0
export LOCAL_RANK=0
export MASTER_ADDR=127.0.0.1
export MASTER_PORT=29314
```

## Task Lists

### `eval_all.sh`

Includes these 11 standard tasks:

```text
auxsolidmath_easy
chartqa100
geometry3k
babyvision
illusionbench_arshia_test
mmsi
phyx_simple
realunify
uni_mmmu
vsp
VisualPuzzles
```

### `eval_all_cot.sh`

Includes these 11 CoT / Visual CoT tasks:

```text
auxsolidmath_easy_visual_cot
chartqa100_visual_cot
geometry3k_visual_cot
babyvision_cot
illusionbench_arshia_visual_cot_split
mmsi_cot
phyx_cot
realunify_cot
uni_mmmu_cot
vsp_cot
VisualPuzzles_visual_cot
```

## Output Layout

The default output directory is:

```text
logs/<model>/
```

Each task writes into its own subdirectory:

```text
logs/<model>/
├── <task_1>/
├── <task_2>/
├── ...
├── summary.json
└── benchmark_summary.json
```

`lmms_eval` itself usually creates an additional timestamped or run-specific
subdirectory under each task directory, so the final `results*.json` files are
often nested deeper. The scripts scan those result files, print summary tables
to the terminal, and also write:

```text
logs/<model>/summary.json
logs/<model>/benchmark_summary.json
```

`summary.json` keeps the main task metrics and filters out `alias` and `stderr`
fields.

`benchmark_summary.json` adds:

- `overall`: sample-weighted mean across all 30 fine-grained subtasks
- `category_overall`: sample-weighted mean within each benchmark category
- `fine_grained`: the chosen metric, score, sample count, and category for each
  fine-grained subtask

## Notes

- Prefer running from the repository root: `bash script/eval_all.sh ...`
- If you start from another working directory, `./logs` is created relative to
  that directory
- The scripts do not currently support resume; if one task fails, rerun the
  full script or edit the `TASKS` array manually
- CoT models are usually more sensitive to `model_args`, especially for
  intermediate result saving and image output settings, so it is safer to
  validate a single task first

## Suggested Workflow

To avoid discovering a bad model configuration halfway through a full run:

1. Run one task first with the same `--model` and `--model_args`.
2. Confirm the output directory, sample logs, and final metrics look correct.
3. Run the full script.
