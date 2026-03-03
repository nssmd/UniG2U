#!/usr/bin/env python3
"""Aggregate lmms-eval outputs into flat, category, and benchmark summaries."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, Iterable, List


FINE_GRAINED_STANDARD = [
    {
        "name": "realunify_attentional_focusing",
        "category": "Real-world Applications",
        "components": [
            {
                "task": "realunify_attentional_focusing",
                "metric": "attentional_focusing",
                "samples": 100,
            }
        ],
    },
    {
        "name": "visual_shortest_path",
        "category": "Real-world Applications",
        "components": [
            {
                "task": "vsp_google_map",
                "metric": "gmap_acc",
                "samples": 50,
            },
            {
                "task": "vsp_collision",
                "metric": "collision_acc",
                "samples": 50,
            },
        ],
    },
    {
        "name": "geometry3k",
        "category": "Geometry Reasoning",
        "components": [
            {
                "task": "geometry3k",
                "metric": "geometry3k_accuracy",
                "samples": 100,
            }
        ],
    },
    {
        "name": "auxsolidmath_easy",
        "category": "Geometry Reasoning",
        "components": [
            {
                "task": "auxsolidmath_easy",
                "metric": "auxsolidmath_text_acc",
                "samples": 100,
            }
        ],
    },
    {
        "name": "phyx_mechanics100",
        "category": "Physics Reasoning",
        "components": [
            {
                "task": "phyx_mechanics100",
                "metric": "eval_results",
                "samples": 100,
            }
        ],
    },
    {
        "name": "phyx_optics100",
        "category": "Physics Reasoning",
        "components": [
            {
                "task": "phyx_optics100",
                "metric": "eval_results",
                "samples": 100,
            }
        ],
    },
    {
        "name": "realunify_mental_reconstruction",
        "category": "Puzzles and Games",
        "components": [
            {
                "task": "realunify_mental_reconstruction",
                "metric": "mental_reconstruction",
                "samples": 100,
            }
        ],
    },
    {
        "name": "realunify_mental_tracking",
        "category": "Puzzles and Games",
        "components": [
            {
                "task": "realunify_mental_tracking",
                "metric": "mental_tracking",
                "samples": 100,
            }
        ],
    },
    {
        "name": "babyvision_visual_tracking",
        "category": "Puzzles and Games",
        "components": [
            {
                "task": "babyvision_visual_tracking",
                "metric": "Visual Tracking",
                "samples": 83,
            }
        ],
    },
    {
        "name": "uni_mmmu_maze100",
        "category": "Puzzles and Games",
        "components": [
            {
                "task": "uni_mmmu_maze100",
                "metric": "maze_text_frame_acc",
                "samples": 100,
            }
        ],
    },
    {
        "name": "uni_mmmu_jigsaw100",
        "category": "Puzzles and Games",
        "components": [
            {
                "task": "uni_mmmu_jigsaw100",
                "metric": "jigsaw_text_acc",
                "samples": 100,
            }
        ],
    },
    {
        "name": "uni_mmmu_sliding54",
        "category": "Puzzles and Games",
        "components": [
            {
                "task": "uni_mmmu_sliding54",
                "metric": "sliding_text_frame_acc",
                "samples": 54,
            }
        ],
    },
    {
        "name": "chartqa100",
        "category": "Chart & Table Reasoning",
        "components": [
            {
                "task": "chartqa100",
                "metric": "relaxed_overall",
                "samples": 100,
            }
        ],
    },
    {
        "name": "mmsi_msr",
        "category": "Spatial Intelligence",
        "components": [
            {
                "task": "mmsi_msr",
                "metric": "MSR",
                "samples": 100,
            }
        ],
    },
    {
        "name": "mmsi_attribute_meas",
        "category": "Spatial Intelligence",
        "components": [
            {
                "task": "mmsi_attribute_meas",
                "metric": "Attribute (Meas.)",
                "samples": 100,
            }
        ],
    },
    {
        "name": "mmsi_attribute_appr",
        "category": "Spatial Intelligence",
        "components": [
            {
                "task": "mmsi_attribute_appr",
                "metric": "Attribute (Appr.)",
                "samples": 100,
            }
        ],
    },
    {
        "name": "mmsi_motion_cam",
        "category": "Spatial Intelligence",
        "components": [
            {
                "task": "mmsi_motion_cam",
                "metric": "Motion (Cam.)",
                "samples": 100,
            }
        ],
    },
    {
        "name": "mmsi_motion_obj",
        "category": "Spatial Intelligence",
        "components": [
            {
                "task": "mmsi_motion_obj",
                "metric": "Motion (Obj.)",
                "samples": 100,
            }
        ],
    },
    {
        "name": "illusionbench_arshia_icon_scene_test",
        "category": "Perception Reasoning",
        "components": [
            {
                "task": "illusionbench_arshia_icon_scene_test",
                "metric": "scene_recall",
                "samples": 100,
            }
        ],
    },
    {
        "name": "illusionbench_arshia_icon_shape_test",
        "category": "Perception Reasoning",
        "components": [
            {
                "task": "illusionbench_arshia_icon_shape_test",
                "metric": "shape_recall",
                "samples": 100,
            }
        ],
    },
    {
        "name": "illusionbench_arshia_in_scene_test",
        "category": "Perception Reasoning",
        "components": [
            {
                "task": "illusionbench_arshia_in_scene_test",
                "metric": "scene_recall",
                "samples": 100,
            }
        ],
    },
    {
        "name": "illusionbench_arshia_in_shape_test",
        "category": "Perception Reasoning",
        "components": [
            {
                "task": "illusionbench_arshia_in_shape_test",
                "metric": "shape_recall",
                "samples": 100,
            }
        ],
    },
    {
        "name": "illusionbench_arshia_logo_scene_test",
        "category": "Perception Reasoning",
        "components": [
            {
                "task": "illusionbench_arshia_logo_scene_test",
                "metric": "scene_recall",
                "samples": 100,
            }
        ],
    },
    {
        "name": "illusionbench_arshia_logo_shape_test",
        "category": "Perception Reasoning",
        "components": [
            {
                "task": "illusionbench_arshia_logo_shape_test",
                "metric": "shape_recall",
                "samples": 100,
            }
        ],
    },
    {
        "name": "VisualPuzzles_algorithmic",
        "category": "Perception Reasoning",
        "components": [
            {
                "task": "VisualPuzzles_algorithmic",
                "metric": "accuracy",
                "samples": 100,
            }
        ],
    },
    {
        "name": "VisualPuzzles_deductive",
        "category": "Perception Reasoning",
        "components": [
            {
                "task": "VisualPuzzles_deductive",
                "metric": "accuracy",
                "samples": 100,
            }
        ],
    },
    {
        "name": "VisualPuzzles_spatial",
        "category": "Perception Reasoning",
        "components": [
            {
                "task": "VisualPuzzles_spatial",
                "metric": "accuracy",
                "samples": 100,
            }
        ],
    },
    {
        "name": "VisualPuzzles_analogical",
        "category": "Perception Reasoning",
        "components": [
            {
                "task": "VisualPuzzles_analogical",
                "metric": "accuracy",
                "samples": 100,
            }
        ],
    },
    {
        "name": "VisualPuzzles_inductive",
        "category": "Perception Reasoning",
        "components": [
            {
                "task": "VisualPuzzles_inductive",
                "metric": "accuracy",
                "samples": 100,
            }
        ],
    },
    {
        "name": "babyvision_fine_grained",
        "category": "Perception Reasoning",
        "components": [
            {
                "task": "babyvision_fine_grained",
                "metric": "Fine-grained Discrimination",
                "samples": 163,
            }
        ],
    },
]


def to_cot_task_name(task_name: str) -> str:
    if task_name.startswith("illusionbench_arshia_") and task_name.endswith("_test"):
        return task_name[:-5] + "_visual_cot"
    if task_name.startswith("VisualPuzzles_"):
        return task_name + "_visual_cot"
    if task_name == "babyvision_fine_grained":
        return "babyvision_fine_grained_visual_cot"
    if task_name == "babyvision_visual_tracking":
        return "babyvision_visual_tracking_visual_cot"
    if task_name.startswith("uni_mmmu_"):
        return task_name + "_visual_cot"
    if task_name.startswith("realunify_"):
        return task_name + "_visual_cot"
    if task_name.startswith("mmsi_"):
        return task_name + "_visual_cot"
    if task_name.startswith("phyx_"):
        return task_name + "_visual_cot"
    return task_name + "_visual_cot"


def get_fine_grained_spec(mode: str) -> List[Dict[str, object]]:
    if mode == "standard":
        return FINE_GRAINED_STANDARD
    if mode == "cot":
        cot_spec = []
        for item in FINE_GRAINED_STANDARD:
            cot_item = dict(item)
            cot_item["components"] = []
            for component in item["components"]:
                cot_component = dict(component)
                cot_component["task"] = to_cot_task_name(str(component["task"]))
                cot_item["components"].append(cot_component)
            cot_spec.append(cot_item)
        return cot_spec
    raise ValueError(f"Unsupported mode: {mode}")


def find_result_files(output_base: Path) -> List[Path]:
    patterns = ("*/*/results*.json", "*/*/*results*.json")
    files: List[Path] = []
    for pattern in patterns:
        files.extend(output_base.glob(pattern))
    return sorted(set(files))


def load_flat_task_metrics(output_base: Path) -> Dict[str, Dict[str, float]]:
    task_metrics: Dict[str, Dict[str, float]] = {}
    for result_file in find_result_files(output_base):
        with result_file.open(encoding="utf-8") as handle:
            payload = json.load(handle)
        for task_name, metrics in payload.get("results", {}).items():
            filtered = task_metrics.setdefault(task_name, {})
            for key, value in metrics.items():
                if key in ("alias", " ") or "stderr" in key:
                    continue
                filtered[key] = value
    return task_metrics


def select_metric(metrics: Dict[str, float], metric_name: str, task_name: str) -> float:
    exact_key = metric_name if metric_name in metrics else None
    if exact_key is not None:
        return metrics[exact_key]

    prefix = f"{metric_name},"
    matched_keys = [key for key in metrics if key.startswith(prefix)]
    if not matched_keys:
        available = ", ".join(sorted(metrics.keys()))
        raise KeyError(f"Metric '{metric_name}' not found for task '{task_name}'. Available keys: {available}")
    if len(matched_keys) > 1:
        raise KeyError(f"Metric '{metric_name}' is ambiguous for task '{task_name}': {matched_keys}")
    return metrics[matched_keys[0]]


def weighted_mean(rows: Iterable[Dict[str, object]]) -> float:
    total_weight = 0
    weighted_sum = 0.0
    for row in rows:
        samples = int(row["samples"])
        value = float(row["value"])
        total_weight += samples
        weighted_sum += samples * value
    if total_weight == 0:
        raise ValueError("Cannot compute weighted mean with zero total weight")
    return weighted_sum / total_weight


def build_benchmark_summary(task_metrics: Dict[str, Dict[str, float]], mode: str) -> Dict[str, object]:
    fine_grained_rows = []
    for spec in get_fine_grained_spec(mode):
        component_rows = []
        for component in spec["components"]:
            task_name = str(component["task"])
            metric_name = str(component["metric"])
            if task_name not in task_metrics:
                raise KeyError(f"Required fine-grained task '{task_name}' not found in loaded results")
            value = select_metric(task_metrics[task_name], metric_name, task_name)
            component_rows.append(
                {
                    "task": task_name,
                    "metric": metric_name,
                    "value": float(value),
                    "samples": int(component["samples"]),
                }
            )
        metric_label = component_rows[0]["metric"]
        if len(component_rows) > 1:
            metric_label = "sample_weighted(" + ", ".join(str(row["metric"]) for row in component_rows) + ")"
        fine_grained_rows.append(
            {
                "task": str(spec["name"]),
                "metric": metric_label,
                "value": weighted_mean(component_rows),
                "samples": sum(int(row["samples"]) for row in component_rows),
                "category": str(spec["category"]),
                "source_tasks": component_rows,
            }
        )

    category_rows: Dict[str, List[Dict[str, object]]] = {}
    for row in fine_grained_rows:
        category_rows.setdefault(str(row["category"]), []).append(row)

    category_overall = {}
    category_samples = {}
    for category, rows in sorted(category_rows.items()):
        category_overall[category] = weighted_mean(rows)
        category_samples[category] = sum(int(row["samples"]) for row in rows)

    overall = weighted_mean(fine_grained_rows)
    total_samples = sum(int(row["samples"]) for row in fine_grained_rows)

    fine_grained_summary = {
        row["task"]: {
            "category": row["category"],
            "metric": row["metric"],
            "value": row["value"],
            "samples": row["samples"],
            "source_tasks": row["source_tasks"],
        }
        for row in fine_grained_rows
    }

    return {
        "mode": mode,
        "aggregation": {
            "fine_grained": "sample_weighted_mean",
            "category_overall": "sample_weighted_mean_over_fine_grained_subtasks",
            "overall": "sample_weighted_mean_over_all_30_fine_grained_subtasks",
        },
        "overall": overall,
        "overall_samples": total_samples,
        "category_overall": category_overall,
        "category_samples": category_samples,
        "fine_grained": fine_grained_summary,
        "task_metrics": task_metrics,
    }


def print_flat_task_table(task_metrics: Dict[str, Dict[str, float]]) -> None:
    print(f"{'Task':<60s} {'Metric':<40s} {'Value'}")
    print("-" * 110)
    for task_name, metrics in sorted(task_metrics.items()):
        for metric_name, value in metrics.items():
            if isinstance(value, float):
                print(f"{task_name:<60s} {metric_name:<40s} {value:.4f}")
            else:
                print(f"{task_name:<60s} {metric_name:<40s} {value}")


def print_benchmark_summary(summary: Dict[str, object]) -> None:
    print("")
    print("BENCHMARK SUMMARY")
    print("-" * 110)
    print(f"{'Overall':<60s} {'sample_weighted':<40s} {summary['overall']:.4f}")
    print("")
    print(f"{'Category':<60s} {'Samples':<10s} {'Value'}")
    print("-" * 110)
    for category, value in summary["category_overall"].items():
        samples = summary["category_samples"][category]
        print(f"{category:<60s} {samples:<10d} {value:.4f}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-base", required=True, help="Base output directory, for example ./logs/qwen2_5_vl")
    parser.add_argument("--mode", choices=("standard", "cot"), required=True, help="Aggregation mode for task name mapping")
    args = parser.parse_args()

    output_base = Path(args.output_base)
    output_base.mkdir(parents=True, exist_ok=True)

    task_metrics = load_flat_task_metrics(output_base)
    benchmark_summary = build_benchmark_summary(task_metrics, args.mode)

    summary_path = output_base / "summary.json"
    summary_path.write_text(json.dumps(task_metrics, indent=2), encoding="utf-8")

    benchmark_summary_path = output_base / "benchmark_summary.json"
    benchmark_summary_path.write_text(json.dumps(benchmark_summary, indent=2), encoding="utf-8")

    print_flat_task_table(task_metrics)
    print("")
    print(f"Summary saved to {summary_path}")
    print(f"Benchmark summary saved to {benchmark_summary_path}")
    print_benchmark_summary(benchmark_summary)


if __name__ == "__main__":
    main()
