<p align="center" width="70%">
<img src="https://i.postimg.cc/KvkLzbF9/WX20241212-014400-2x.png">
</p>

# The Evaluation Suite of Large Multimodal Models 

> UniG2U is built on top of `lmms-eval` and extends it with additional
> benchmark tasks, model integrations, Visual CoT pipelines, and one-shot
> evaluation scripts for our benchmark workflow.

UniG2U repository: https://github.com/nssmd/UniG2U.git  
Upstream `lmms-eval`: https://github.com/EvolvingLMMs-Lab/lmms-eval

The sections below retain the original `lmms-eval` project overview and usage
context, followed by UniG2U-specific additions.

[![PyPI](https://img.shields.io/pypi/v/lmms-eval)](https://pypi.org/project/lmms-eval)
![PyPI - Downloads](https://img.shields.io/pypi/dm/lmms-eval)
[![GitHub contributors](https://img.shields.io/github/contributors/EvolvingLMMs-Lab/lmms-eval)](https://github.com/EvolvingLMMs-Lab/lmms-eval/graphs/contributors)
[![issue resolution](https://img.shields.io/github/issues-closed-raw/EvolvingLMMs-Lab/lmms-eval)](https://github.com/EvolvingLMMs-Lab/lmms-eval/issues)
[![open issues](https://img.shields.io/github/issues-raw/EvolvingLMMs-Lab/lmms-eval)](https://github.com/EvolvingLMMs-Lab/lmms-eval/issues)

> Accelerating the development of large multimodal models (LMMs) with `lmms-eval`. We support most text, image, video and audio tasks.

🏠 [LMMs-Lab Homepage](https://www.lmms-lab.com/) | 🤗 [Huggingface Datasets](https://huggingface.co/lmms-lab) | <a href="https://emoji.gg/emoji/1684-discord-thread"><img src="https://cdn3.emoji.gg/emojis/1684-discord-thread.png" width="14px" height="14px" alt="Discord_Thread"></a> [discord/lmms-eval](https://discord.gg/zdkwKUqrPy)

📖 [Supported Tasks (100+)](https://github.com/EvolvingLMMs-Lab/lmms-eval/blob/main/docs/current_tasks.md) | 🌟 [Supported Models (30+)](https://github.com/EvolvingLMMs-Lab/lmms-eval/tree/main/lmms_eval/models) | 📚 [Documentation](docs/README.md)

---

## Annoucement

- [2025-10] 🚀🚀 **LMMs-Eval v0.5** is here! This major release introduces comprehensive audio evaluation, response caching, 5 new models (GPT-4o Audio Preview, Gemma-3, LongViLA-R1, LLaVA-OneVision 1.5, Thyme), and 50+ new benchmark variants spanning audio (Step2, VoiceBench, WenetSpeech), vision (CharXiv, Lemonade), and reasoning (CSBench, SciBench, MedQA, SuperGPQA) with reproducible results. Please refer to the [release notes](https://github.com/EvolvingLMMs-Lab/lmms-eval/blob/main/docs/lmms-eval-0.5.md) for details.
- [2025-07] 🚀🚀 We have released the `lmms-eval-0.4`. Please refer to the [release notes](https://github.com/EvolvingLMMs-Lab/lmms-eval/blob/main/docs/lmms-eval-0.4.md) for more details. This is a major update with new features and improvements, for users wish to use `lmms-eval-0.3` please refer to the branch `stable/v0d3`. For our mission to better reproductability, we've opened a specific thread to discuss about the model's eval results in [discussion](https://github.com/EvolvingLMMs-Lab/lmms-eval/discussions/779).
- [2025-07] 🎉🎉 We welcome the new task [PhyX](https://phyx-bench.github.io/), the first large-scale benchmark designed to assess models capacity for physics-grounded reasoning in visual scenarios.
- [2025-06] 🎉🎉 We welcome the new task [VideoMathQA](https://mbzuai-oryx.github.io/VideoMathQA), designed to evaluate mathematical reasoning in real-world educational videos.
- [2025-04] 🚀🚀 Introducing [Aero-1-Audio](https://www.lmms-lab.com/posts/aero_audio/) — a compact yet mighty audio model. We have officially supports evaluation for Aero-1-Audio and it supports batched evaluations! Feel free to try out.
- [2025-02] 🚀🚀 We have integrated [`vllm`](https://github.com/EvolvingLMMs-Lab/lmms-eval/pull/544) into our models, enabling accelerated evaluation for both multimodal and language models. Additionally, we have incorporated [`openai_compatible`](https://github.com/EvolvingLMMs-Lab/lmms-eval/pull/546) to support the evaluation of any API-based model that follows the OpenAI API format. Check the usages [here](https://github.com/EvolvingLMMs-Lab/lmms-eval/tree/main/miscs/model_dryruns).

<details>
<summary>Below is a chronological list of recent tasks, models, and features added by our amazing contributors. </summary>

- [2025-01] 🎓🎓 We have released our new benchmark: [Video-MMMU: Evaluating Knowledge Acquisition from Multi-Discipline Professional Videos](https://arxiv.org/abs/2501.13826). Please refer to the [project page](https://videommmu.github.io/) for more details.
- [2024-12] 🎉🎉 We have presented [MME-Survey: A Comprehensive Survey on Evaluation of Multimodal LLMs](https://arxiv.org/pdf/2411.15296), jointly with [MME Team](https://github.com/BradyFU/Video-MME) and [OpenCompass Team](https://github.com/open-compass).
- [2024-11] 🔈🔊 The `lmms-eval/v0.3.0` has been upgraded to support audio evaluations for audio models like Qwen2-Audio and Gemini-Audio across tasks such as AIR-Bench, Clotho-AQA, LibriSpeech, and more. Please refer to the [blog](https://github.com/EvolvingLMMs-Lab/lmms-eval/blob/main/docs/lmms-eval-0.3.md) for more details!
- [2024-10] 🎉🎉 We welcome the new task [NaturalBench](https://huggingface.co/datasets/BaiqiL/NaturalBench), a vision-centric VQA benchmark (NeurIPS'24) that challenges vision-language models with simple questions about natural imagery.
- [2024-10] 🎉🎉 We welcome the new task [TemporalBench](https://huggingface.co/datasets/microsoft/TemporalBench) for fine-grained temporal understanding and reasoning for videos, which reveals a huge (>30%) human-AI gap.
- [2024-10] 🎉🎉 We welcome the new tasks [VDC](https://rese1f.github.io/aurora-web/) for video detailed captioning, [MovieChat-1K](https://rese1f.github.io/MovieChat/) for long-form video understanding, and [Vinoground](https://vinoground.github.io/), a temporal counterfactual LMM benchmark composed of 1000 short natural video-caption pairs. We also welcome the new models: [AuroraCap](https://github.com/rese1f/aurora) and [MovieChat](https://github.com/rese1f/MovieChat).
- [2024-09] 🎉🎉 We welcome the new tasks [MMSearch](https://mmsearch.github.io/) and [MME-RealWorld](https://mme-realworld.github.io/) for inference acceleration
- [2024-09] ⚙️️⚙️️️️ We upgrade `lmms-eval` to `0.2.3` with more tasks and features. We support a compact set of language tasks evaluations (code credit to [lm-evaluation-harness](https://github.com/EleutherAI/lm-evaluation-harness)), and we remove the registration logic at start (for all models and tasks) to reduce the overhead. Now `lmms-eval` only launches necessary tasks/models. Please check the [release notes](https://github.com/EvolvingLMMs-Lab/lmms-eval/releases/tag/v0.2.3) for more details.
- [2024-08] 🎉🎉 We welcome the new model [LLaVA-OneVision](https://huggingface.co/papers/2408.03326), [Mantis](https://github.com/EvolvingLMMs-Lab/lmms-eval/pull/162), new tasks [MVBench](https://huggingface.co/datasets/OpenGVLab/MVBench), [LongVideoBench](https://github.com/EvolvingLMMs-Lab/lmms-eval/pull/117), [MMStar](https://github.com/EvolvingLMMs-Lab/lmms-eval/pull/158). We provide new feature of SGlang Runtime API for llava-onevision model, please refer the [doc](https://github.com/EvolvingLMMs-Lab/lmms-eval/blob/main/docs/commands.md) for inference acceleration
- [2024-07] 👨‍💻👨‍💻 The `lmms-eval/v0.2.1` has been upgraded to support more models, including [LongVA](https://github.com/EvolvingLMMs-Lab/LongVA), [InternVL-2](https://github.com/OpenGVLab/InternVL), [VILA](https://github.com/NVlabs/VILA), and many more evaluation tasks, e.g. [Details Captions](https://github.com/EvolvingLMMs-Lab/lmms-eval/pull/136), [MLVU](https://arxiv.org/abs/2406.04264), [WildVision-Bench](https://huggingface.co/datasets/WildVision/wildvision-arena-data), [VITATECS](https://github.com/lscpku/VITATECS) and [LLaVA-Interleave-Bench](https://llava-vl.github.io/blog/2024-06-16-llava-next-interleave/).
- [2024-07] 🎉🎉 We have released the [technical report](https://arxiv.org/abs/2407.12772) and [LiveBench](https://huggingface.co/spaces/lmms-lab/LiveBench)! 
- [2024-06] 🎬🎬 The `lmms-eval/v0.2.0` has been upgraded to support video evaluations for video models like LLaVA-NeXT Video and Gemini 1.5 Pro across tasks such as EgoSchema, PerceptionTest, VideoMME, and more. Please refer to the [blog](https://lmms-lab.github.io/posts/lmms-eval-0.2/) for more details!
- [2024-03] 📝📝 We have released the first version of `lmms-eval`, please refer to the [blog](https://lmms-lab.github.io/posts/lmms-eval-0.1/) for more details!

</details>

## Why `lmms-eval`?

We're on an exciting journey toward creating Artificial General Intelligence (AGI), much like the enthusiasm of the 1960s moon landing. This journey is powered by advanced large language models (LLMs) and large multimodal models (LMMs), which are complex systems capable of understanding, learning, and performing a wide variety of human tasks.

To gauge how advanced these models are, we use a variety of evaluation benchmarks. These benchmarks are tools that help us understand the capabilities of these models, showing us how close we are to achieving AGI. However, finding and using these benchmarks is a big challenge. The necessary benchmarks and datasets are spread out and hidden in various places like Google Drive, Dropbox, and different school and research lab websites. It feels like we're on a treasure hunt, but the maps are scattered everywhere.

In the field of language models, there has been a valuable precedent set by the work of [lm-evaluation-harness](https://github.com/EleutherAI/lm-evaluation-harness). They offer integrated data and model interfaces, enabling rapid evaluation of language models and serving as the backend support framework for the [open-llm-leaderboard](https://huggingface.co/spaces/HuggingFaceH4/open_llm_leaderboard), and has gradually become the underlying ecosystem of the era of foundation models.

We humbly obsorbed the exquisite and efficient design of [lm-evaluation-harness](https://github.com/EleutherAI/lm-evaluation-harness) and introduce **lmms-eval**, an evaluation framework meticulously crafted for consistent and efficient evaluation of LMM.

## Installation

### Using uv (Recommended for consistent environments)

We use `uv` for package management to ensure all developers use exactly the same package versions. First, install uv:
```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

For development with consistent environment:
```bash
git clone https://github.com/nssmd/UniG2U.git
cd UniG2U
# Recommend
uv pip install -e ".[all]"
# If you want to use uv sync
# uv sync  # This creates/updates your environment from uv.lock
```

To run commands:
```bash
uv run python -m lmms_eval --help  # Run any command with uv run
```

To add new dependencies:
```bash
uv add <package>  # Updates both pyproject.toml and uv.lock
```

### Alternative Installation

For direct usage from Git:
```bash
uv venv eval
uv venv --python 3.12
source eval/bin/activate
# You might need to add and include your own task yaml if using this installation
uv pip install git+https://github.com/nssmd/UniG2U.git
```

<details>
<summary>Reproduction of LLaVA-1.5's paper results</summary>

You can check the [environment install script](miscs/repr_scripts.sh) and [torch environment info](miscs/repr_torch_envs.txt) to **reproduce LLaVA-1.5's paper results**. We found torch/cuda versions difference would cause small variations in the results, we provide the [results check](miscs/llava_result_check.md) with different environments.

</details>

If you want to test on caption dataset such as `coco`, `refcoco`, and `nocaps`, you will need to have `java==1.8.0` to let pycocoeval api to work. If you don't have it, you can install by using conda
```
conda install openjdk=8
```
you can then check your java version by `java -version` 


<details>
<summary>Comprehensive Evaluation Results of LLaVA Family Models</summary>
<br>

As demonstrated by the extensive table below, we aim to provide detailed information for readers to understand the datasets included in lmms-eval and some specific details about these datasets (we remain grateful for any corrections readers may have during our evaluation process).

We provide a Google Sheet for the detailed results of the LLaVA series models on different datasets. You can access the sheet [here](https://docs.google.com/spreadsheets/d/1a5ImfdKATDI8T7Cwh6eH-bEsnQFzanFraFUgcS9KHWc/edit?usp=sharing). It's a live sheet, and we are updating it with new results.

<p align="center" width="100%">
<img src="https://i.postimg.cc/jdw497NS/WX20240307-162526-2x.png"  width="100%" height="80%">
</p>

We also provide the raw data exported from Weights & Biases for the detailed results of the LLaVA series models on different datasets. You can access the raw data [here](https://docs.google.com/spreadsheets/d/1AvaEmuG4csSmXaHjgu4ei1KBMmNNW8wflOD_kkTDdv8/edit?usp=sharing).

</details>
<br>

If you want to test [VILA](https://github.com/NVlabs/VILA), you should install the following dependencies:

```bash
pip install s2wrapper@git+https://github.com/bfshi/scaling_on_scales
```

Our Development will be continuing on the main branch, and we encourage you to give us feedback on what features are desired and how to improve the library further, or ask questions, either in issues or PRs on GitHub.

## Usages

> More examples can be found in [examples/models](examples/models)

**Evaluation of OpenAI-Compatible Model**

```bash
bash examples/models/openai_compatible.sh
bash examples/models/xai_grok.sh
```

**Evaluation of vLLM**

```bash
bash examples/models/vllm_qwen2vl.sh
```

**Evaluation of LLaVA-OneVision**

```bash
bash examples/models/llava_onevision.sh
```

**Evaluation of LLaVA-OneVision1_5**

```bash
bash examples/models/llava_onevision1_5.sh
```

**Evaluation of LLaMA-3.2-Vision**

```bash
bash examples/models/llama_vision.sh
```

**Evaluation of Qwen2-VL**

```bash
bash examples/models/qwen2_vl.sh
bash examples/models/qwen2_5_vl.sh
```

**Evaluation of LLaVA on MME**

If you want to test LLaVA 1.5, you will have to clone their repo from [LLaVA](https://github.com/haotian-liu/LLaVA) and

```bash
bash examples/models/llava_next.sh
```

**Evaluation with tensor parallel for bigger model (llava-next-72b)**

```bash
bash examples/models/tensor_parallel.sh
```

**Evaluation with SGLang for bigger model (llava-next-72b)**

```bash
bash examples/models/sglang.sh
```

**Evaluation with vLLM for bigger model (llava-next-72b)**

```bash
bash examples/models/vllm_qwen2vl.sh
```

**More Parameters**

```bash
python3 -m lmms_eval --help
```

**Environmental Variables**
Before running experiments and evaluations, we recommend you to export following environment variables to your environment. Some are necessary for certain tasks to run.

```bash
export OPENAI_API_KEY="<YOUR_API_KEY>"
export HF_HOME="<Path to HF cache>" 
export HF_TOKEN="<YOUR_API_KEY>"
export HF_HUB_ENABLE_HF_TRANSFER="1"
export REKA_API_KEY="<YOUR_API_KEY>"
# Other possible environment variables include 
# ANTHROPIC_API_KEY,DASHSCOPE_API_KEY etc.
```

**Common Environment Issues**

Sometimes you might encounter some common issues for example error related to httpx or protobuf. To solve these issues, you can first try

```bash
python3 -m pip install httpx==0.23.3;
python3 -m pip install protobuf==3.20;
# If you are using numpy==2.x, sometimes may causing errors
python3 -m pip install numpy==1.26;
# Someties sentencepiece are required for tokenizer to work
python3 -m pip install sentencepiece;
```

## UniG2U Extensions

This repository is based on
[`lmms-eval`](https://github.com/EvolvingLMMs-Lab/lmms-eval) and extends it for
our benchmark and model evaluation workflow.

Compared with upstream `lmms-eval`, UniG2U adds:

- Custom benchmark tasks under `lmms_eval/tasks`, including the UniG2U suite,
  benchmark wrappers, and prompt / scoring utilities for our task variants
- Additional model integrations under `lmms_eval/models`, including custom
  multimodal backends and Visual CoT style model wrappers
- One-shot benchmark scripts under `script/` for running the full UniG2U task
  suite and generating final aggregate reports automatically
- Benchmark aggregation logic in `script/aggregate_results.py` for computing:
  - task summaries
  - category-level scores
  - benchmark-level `overall`

The UniG2U benchmark coverage in this repository includes task families such as
AuxSolidMath-Easy, ChartQA, Geometry3K, BabyVision, IllusionBench, MMSI-Bench,
PhyX, RealUnify, Uni-MMMU, VSP, and VisualPuzzles, together with their CoT /
Visual CoT variants where applicable.

Our added or customized model integrations include examples such as
`uniworld`, `uniworld_visual_cot`, `emu3`, `emu3_visual_cot`, `mio`,
`qwen_image_edit`, and `qwen_image_edit_visual_cot`, along with several other
Visual CoT wrappers registered in `lmms_eval/models`.

The main directories to know are:

- `lmms_eval/tasks/`: benchmark task definitions, YAML configs, prompt and
  metric utilities
- `lmms_eval/models/`: model registry plus our custom model implementations
- `script/eval_all.sh`: one-shot standard benchmark runner
- `script/eval_all_cot.sh`: one-shot CoT / Visual CoT benchmark runner
- `script/aggregate_results.py`: post-processing and benchmark aggregation
- `script/README.md`: focused documentation for the one-shot scripts

Some of our custom models depend on external repositories or model-specific
code. For example:

- `uniworld` / `uniworld_visual_cot` expect the UniWorld codebase under
  `UniWorld/UniWorld-V1`
- `mio` expects the MIO repository under `MIO/`

If you use those models, make sure their extra dependencies and external code
are prepared before running evaluation.

## UniG2U Benchmark Usage

The easiest way to run our benchmark is through the scripts in `script/`.
These scripts provide a stable interface on top of `lmms_eval` and run the full
task suite sequentially.

Test script paths:

- `script/eval_all.sh`
- `script/eval_all_cot.sh`

Recommended usage from the repository root:

```bash
bash script/eval_all.sh \
  --model qwen2_5_vl \
  --model_args "pretrained=Qwen/Qwen2.5-VL-3B-Instruct"
```

For CoT / Visual CoT models:

```bash
bash script/eval_all_cot.sh \
  --model bagel_visual_cot \
  --model_args "pretrained=ByteDance-Seed/BAGEL-7B-MoT,save_intermediate=true"
```

Both scripts:

- run the predefined UniG2U task list sequentially
- always use `--batch_size 1`
- always enable `--log_samples`
- stop on the first failed task
- automatically aggregate final results after all tasks finish

Under the hood, each task is executed through:

```bash
uv run python -m lmms_eval \
  --model <model_name> \
  --model_args "<model_args>" \
  --tasks <task_name> \
  --batch_size 1 \
  --log_samples \
  --output_path ./logs/<model_name>/<task_name>
```

## UniG2U Runtime Notes

Before running the UniG2U scripts, make sure:

- `uv` is installed and available in your shell
- this repository has been installed with `uv pip install -e ".[all]"`
- the selected model name is registered in `lmms_eval/models`
- model weights and datasets needed by that task are available in your local
  runtime

For local Hugging Face models on GPU, a common pattern is:

```bash
uv run python -m lmms_eval \
  --model qwen2_5_vl \
  --model_args "pretrained=Qwen/Qwen2.5-VL-3B-Instruct,device_map=auto"
```

## UniG2U Outputs

By default, benchmark runs are written under:

```text
logs/<model_name>/
```

After the scripts finish, UniG2U writes:

- `logs/<model_name>/summary.json`: task-level summary extracted from raw
  `results*.json`
- `logs/<model_name>/benchmark_summary.json`: benchmark-level aggregation with
  `overall`, `category_overall`, and fine-grained metric details

`benchmark_summary.json` is the main file for reporting final UniG2U results.
It includes:

- `overall`: sample-weighted score across the full benchmark
- `category_overall`: sample-weighted score for each category
- `fine_grained`: selected metric, score, sample count, and category for each
  fine-grained subtask

For more details on the one-shot scripts, task lists, and output layout, see
`script/README.md`.

## Add Customized Model and Dataset

Please refer to our [documentation](docs/README.md).

## Acknowledgement

lmms_eval is a fork of [lm-eval-harness](https://github.com/EleutherAI/lm-evaluation-harness). We recommend you to read through the [docs of lm-eval-harness](https://github.com/EleutherAI/lm-evaluation-harness/tree/main/docs) for relevant information. 

---

Below are the changes we made to the original API:
- Build context now only pass in idx and process image and doc during the model responding phase. This is due to the fact that dataset now contains lots of images and we can't store them in the doc like the original lm-eval-harness other wise the cpu memory would explode.
- Instance.args (lmms_eval/api/instance.py) now contains a list of images to be inputted to lmms.
- lm-eval-harness supports all HF language models as single model class. Currently this is not possible of lmms because the input/output format of lmms in HF are not yet unified. Thererfore, we have to create a new class for each lmms model. This is not ideal and we will try to unify them in the future.
---

## Citations

```shell
@misc{zhang2024lmmsevalrealitycheckevaluation,
      title={LMMs-Eval: Reality Check on the Evaluation of Large Multimodal Models}, 
      author={Kaichen Zhang and Bo Li and Peiyuan Zhang and Fanyi Pu and Joshua Adrian Cahyono and Kairui Hu and Shuai Liu and Yuanhan Zhang and Jingkang Yang and Chunyuan Li and Ziwei Liu},
      year={2024},
      eprint={2407.12772},
      archivePrefix={arXiv},
      primaryClass={cs.CL},
      url={https://arxiv.org/abs/2407.12772}, 
}

@misc{lmms_eval2024,
    title={LMMs-Eval: Accelerating the Development of Large Multimoal Models},
    url={https://github.com/EvolvingLMMs-Lab/lmms-eval},
    author={Bo Li*, Peiyuan Zhang*, Kaichen Zhang*, Fanyi Pu*, Xinrun Du, Yuhao Dong, Haotian Liu, Yuanhan Zhang, Ge Zhang, Chunyuan Li and Ziwei Liu},
    publisher    = {Zenodo},
    version      = {v0.1.0},
    month={March},
    year={2024}
}
```
