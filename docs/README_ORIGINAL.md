<p align="center">
  <img src="figures/logo.png" width="20%">
</p>

<h1 align="center">LLM-Pruner</h1>

<p align="center">
  <a href="https://opensource.org/licenses/Apache-2.0">
    <img alt="License: Apache 2.0" src="https://img.shields.io/badge/License-Apache%202.0-4E94CE.svg">
  </a>
</p>

<h3 align="center">On the Structural Pruning of Large Language Models</h3>

> **[LLM-Pruner: On the Structural Pruning of Large Language Models](https://arxiv.org/abs/2305.11627)** [[arXiv]](https://arxiv.org/abs/2305.11627)  
> *Xinyin Ma, Gongfan Fang, Xinchao Wang* — National University of Singapore

## Overview

LLM-Pruner is a structural pruning framework for large language models. It compresses LLMs to any target size while preserving multi-task capabilities through task-agnostic compression and lightweight post-training.

**Highlights:**
- **Task-agnostic compression**: Retains original multi-task solving abilities
- **Data-efficient recovery**: Recovers performance with only 50k public samples (Alpaca)
- **Fast compression**: ~3 minutes for pruning, ~3 hours for post-training
- **Automatic structural pruning**: Minimal human effort for new LLMs

**Supported Models:** LLaMA, Llama-2, Llama-3/3.1, Vicuna, BLOOM, Baichuan, TinyLlama, ChatGLM

## Quick Start

### Installation

```bash
pip install -r requirement.txt
```

### Minimal Example

```bash
bash script/llama_prune.sh
```

This automatically downloads the LLaMA-7B model and dataset, pruning ~20% of parameters.

## Usage

### 1. Pruning

**LLaMA / Llama-2 (~20% parameters pruned):**

```bash
python hf_prune.py --pruning_ratio 0.25 \
    --block_wise \
    --block_mlp_layer_start 4 --block_mlp_layer_end 30 \
    --block_attention_layer_start 4 --block_attention_layer_end 30 \
    --pruner_type taylor \
    --test_after_train \
    --device cpu --eval_device cuda \
    --save_ckpt_log_name llama_prune
```

**Llama-3 / Llama-3.1:**

```bash
python llama3.py --pruning_ratio 0.25 \
    --device cuda --eval_device cuda \
    --base_model meta-llama/Meta-Llama-3-8B-Instruct \
    --block_wise --block_mlp_layer_start 4 --block_mlp_layer_end 30 \
    --block_attention_layer_start 4 --block_attention_layer_end 30 \
    --save_ckpt_log_name llama3_prune \
    --pruner_type taylor --taylor param_first \
    --max_seq_len 2048 \
    --test_after_train --test_before_train --save_model
```

For other models (Vicuna, BLOOM, Baichuan, etc.), see [examples/README.md](examples/README.md).

Key arguments:
- `--base_model`: Model identifier for `AutoModel.from_pretrained`
- `--pruning_ratio`: Target group pruning ratio
- `--block_wise` / `--channel_wise` / `--layer_wise`: Pruning granularity
- `--pruner_type`: Importance criterion (`taylor`, `l1`, `l2`, `random`)
- `--device` / `--eval_device`: Devices for pruning and evaluation

### 2. Post-Training

```bash
CUDA_VISIBLE_DEVICES=0 python post_training.py \
    --prune_model prune_log/PATH_TO_PRUNE_MODEL/pytorch_model.bin \
    --data_path yahma/alpaca-cleaned \
    --lora_r 8 --num_epochs 2 \
    --learning_rate 1e-4 --batch_size 64 \
    --output_dir tune_log/PATH_TO_SAVE_TUNE_MODEL
```

> **Tip:** Training LLaMA-2 in float16 may produce NaN; use bfloat16 instead.

### 3. Loading Pruned Models

```python
pruned_dict = torch.load(YOUR_CHECKPOINT_PATH, map_location='cpu')
tokenizer, model = pruned_dict['tokenizer'], pruned_dict['model']
```

> Note: Due to heterogeneous layer widths after pruning, `from_pretrained()` is not supported. Use `torch.load` instead.

### 4. Generation

```bash
# Pre-trained
python generate.py --model_type pretrain

# Pruned model
python generate.py --model_type pruneLLM --ckpt <YOUR_PRUNED_MODEL_PATH>

# Pruned + post-trained
python generate.py --model_type tune_prune_LLM \
    --ckpt <YOUR_PRUNED_MODEL_PATH> \
    --lora_ckpt <YOUR_LORA_PATH>
```

### 5. Evaluation

We use [lm-evaluation-harness](https://github.com/EleutherAI/lm-evaluation-harness) for zero-shot evaluation. A simplified script is provided:

```bash
CUDA_VISIBLE_DEVICES=0 bash scripts/evaluate.sh \
    PATH_OR_NAME_TO_BASE_MODEL \
    PATH_TO_SAVE_TUNE_MODEL \
    PATH_TO_PRUNE_MODEL \
    EPOCHS_TO_EVALUATE
```

### 6. Profiling (MACs, Params, Memory)

```bash
python test_speedup.py --model_type pruneLLM --ckpt <YOUR_PRUNED_MODEL_PATH>
```

## Results

Zero-shot evaluation on LLaMA-7B:

| Model | #Param | Memory | Latency | BoolQ | PIQA | HellaSwag | WinoGrande | ARC-e | ARC-c | OBQA | Avg |
|-------|--------|--------|---------|-------|------|-----------|------------|-------|-------|------|-----|
| LLaMA-7B | 6.74B | 12884MiB | 69.32s | 73.18 | 78.35 | 72.99 | 67.01 | 67.45 | 41.38 | 42.40 | 63.25 |
| LLaMA-5.4B (Alpaca 50k) | 5.47B | 10488MiB | 58.55s | 64.62 | 77.20 | 68.80 | 63.14 | 64.31 | 36.77 | 39.80 | 59.23 |
| LLaMA-5.4B (LaMini 2.59M) | 5.47B | 10488MiB | 58.55s | 76.57 | 77.37 | 66.60 | 65.82 | 70.62 | 40.70 | 38.80 | 62.36 |

More results (Llama-3/3.1, TinyLlama, etc.) are available in [more_results/README.md](more_results/README.md).

## Acknowledgements

- Logo generated by [Stable Diffusion](https://dreamstudio.ai/generate)
- Evaluation based on [lm-evaluation-harness](https://github.com/EleutherAI/lm-evaluation-harness)
- LLaMA: [facebookresearch/llama](https://github.com/facebookresearch/llama)
- PEFT: [huggingface/peft](https://github.com/huggingface/peft)

## Citation

```bibtex
@inproceedings{ma2023llmpruner,
  title={LLM-Pruner: On the Structural Pruning of Large Language Models},
  author={Xinyin Ma and Gongfan Fang and Xinchao Wang},
  booktitle={Advances in Neural Information Processing Systems},
  year={2023}
}

@article{fang2023depgraph,
  title={DepGraph: Towards Any Structural Pruning},
  author={Fang, Gongfan and Ma, Xinyin and Song, Mingli and Mi, Michael Bi and Wang, Xinchao},
  journal={The IEEE/CVF Conference on Computer Vision and Pattern Recognition},
  year={2023}
}
```
