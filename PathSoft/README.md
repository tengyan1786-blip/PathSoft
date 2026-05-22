# PathSoft

This repository includes the two-stage code used in our experiments:

1. `pathmlp/`: first-stage candidate path ranking with a frozen GTE encoder and a lightweight MLP scorer.
2. `pathsoft/`: second-stage PathSoftAdapter training and inference with a frozen LLM.


## Repository Contents

```text
pathmlp/     # Stage-1 path ranking: encode, train, and score candidate paths
pathsoft/    # Stage-2 adapter training and inference
README.md    # Reproduction instructions
requirements.txt
```

## What You Need to Prepare

Before running PathSoft, prepare the following files and models:

1. Flat question-path JSONL files for PathMLP training and scoring.
2. Grouped scored-path JSONL files for PathSoftAdapter training and inference, produced by Stage 1.
3. A frozen text encoder for Stage 1, such as `gte-large-en-v1.5`.
4. A frozen BERT encoder for Stage 2 path element encoding, such as `bert-base-uncased`.
5. A frozen LLM backbone, such as `Qwen2.5-7B-Instruct` or `Llama-3.1-8B-Instruct`.

All commands below should be run from the repository root.

## Installation

```bash
pip install -r requirements.txt
```

The code was used with local HuggingFace model paths for GTE, BERT, Qwen2.5-7B-Instruct, and Llama-3.1-8B-Instruct. You can replace them with your own local paths or HuggingFace model names.

## Data Formats

### PathMLP input format

PathMLP expects a flat JSONL file. Each line is one question-path pair:

```json
{
  "qid": "...",
  "question": "...",
  "path_text": "Entity [relation] Entity [relation] Entity",
  "path_edges": [],
  "answers": ["..."],
  "answer_entity_ids": [],
  "end_entity": "...",
  "end_entity_id": 0,
  "label": 1
}
```

`label=1` means the terminal entity of the path matches a gold answer; otherwise `label=0`.

### PathSoft input format

PathSoft expects grouped scored-path JSONL files. Each line is one question with ranked paths:

```json
{
  "qid": "...",
  "question": "...",
  "gold_answers": ["..."],
  "scored_paths": [
    {
      "path_text": "Entity [relation] Entity",
      "score": 0.9,
      "label": 1
    }
  ]
}
```

`path_text` is parsed into entity/relation path elements. `score` is the PathMLP score used for ranking and optional score-guided attention.

## Stage 1: Train PathMLP

### 1. Encode question-path pairs with frozen GTE

```bash
python pathmlp/encode_path_data.py \
  --input /path/to/train_paths_flat.jsonl \
  --output-dir ./encoded/train \
  --model-path /path/to/gte-large-en-v1.5 \
  --batch-size 16 \
  --shard-size 2048 \
  --max-length 128

python pathmlp/encode_path_data.py \
  --input /path/to/val_paths_flat.jsonl \
  --output-dir ./encoded/val \
  --model-path /path/to/gte-large-en-v1.5 \
  --batch-size 16 \
  --shard-size 2048 \
  --max-length 128
```

### 2. Train PathMLP

```bash
python pathmlp/train_path_mlp.py \
  --train-dir ./encoded/train \
  --val-dir ./encoded/val \
  --output-dir ./outputs/pathmlp \
  --epochs 20 \
  --batch-size 128 \
  --lr 1e-3
```

The best checkpoint is saved as:

```text
./outputs/pathmlp/best_model.pth
```

### 3. Score candidate paths

First encode the split to be scored:

```bash
python pathmlp/encode_path_data.py \
  --input /path/to/test_paths_flat.jsonl \
  --output-dir ./encoded/test \
  --model-path /path/to/gte-large-en-v1.5 \
  --batch-size 16 \
  --shard-size 2048 \
  --max-length 128
```

Then score and keep the top paths per question:

```bash
python pathmlp/score_paths_stream.py \
  --encoded-dir ./encoded/test \
  --checkpoint ./outputs/pathmlp/best_model.pth \
  --output ./outputs/test_scored_paths_top200.jsonl \
  --top-k 200 \
  --batch-size 512
```

Repeat this scoring step for train/validation/test splits as needed. The resulting grouped scored-path JSONL files are used by PathSoftAdapter.

## Stage 2: Train PathSoftAdapter

```bash
python pathsoft/train.py \
  --train_path_file /path/to/train_scored_paths_top200.jsonl \
  --eval_path_file /path/to/val_scored_paths_top200.jsonl \
  --cache_dir ./cache \
  --output_dir ./outputs/pathsoft_qwen_k20 \
  --llm_model /path/to/Qwen2.5-7B-Instruct \
  --bert_model bert-base-uncased \
  --llm_dtype float16 \
  --llm_device_map auto \
  --adapter_device cuda:0 \
  --max_paths 20 \
  --max_path_len 7 \
  --hidden_dim 256 \
  --gru_layers 2 \
  --num_heads 4 \
  --n_tokens 16 \
  --dropout 0.1 \
  --epochs 10 \
  --batch_size 2 \
  --lr 1e-4 \
  --weight_decay 0.01 \
  --max_context_tokens 1024 \
  --max_answer_tokens 64
```

For Llama-3.1-8B-Instruct, replace `--llm_model` with the Llama model path. The LLM is frozen; only PathSoftAdapter parameters are updated.

## Stage 2: Inference

```bash
python pathsoft/inference.py \
  --eval_path_file /path/to/test_scored_paths_top200.jsonl \
  --checkpoint ./outputs/pathsoft_qwen_k20/best_adapter.pt \
  --cache_dir ./cache \
  --output_dir ./outputs/pathsoft_qwen_k20_test \
  --llm_model /path/to/Qwen2.5-7B-Instruct \
  --llm_dtype float16 \
  --llm_device_map auto \
  --adapter_device cuda:0 \
  --batch_size 2 \
  --max_paths 20 \
  --max_path_len 7 \
  --max_context_tokens 1024 \
  --max_new_tokens 64
```

Inference writes:

```text
pathsoft_predictions.jsonl
```

Each line contains `qid`, `question`, `gold_answers`, and `raw_prediction`.

