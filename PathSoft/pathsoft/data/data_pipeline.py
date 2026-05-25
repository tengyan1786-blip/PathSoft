import json
import os
import re
from collections import OrderedDict

import torch
from torch.utils.data import Dataset
from transformers import BertModel, BertTokenizer
from tqdm import tqdm


def parse_path_text(path_text):
    parts = re.split(r"\s*\[(.*?)\]\s*", path_text.strip())
    return [part.strip() for part in parts if part and part.strip()]


def format_paths_text(paths):
    lines = []
    for idx, path in enumerate(paths, start=1):
        lines.append(f"{idx}. " + " -> ".join(path))
    return "\n".join(lines)


def load_grouped_scored_paths(path):
    samples = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            row = json.loads(line)
            scored_paths = list(row.get("scored_paths", []))
            scored_paths.sort(
                key=lambda item: float(
                    item.get("score", item.get("weighted_score", item.get("v2_score", 0.0)))
                ),
                reverse=True,
            )
            parsed_paths = [parse_path_text(item["path_text"]) for item in scored_paths]
            path_scores = [
                float(item.get("score", item.get("weighted_score", item.get("v2_score", 0.0))))
                for item in scored_paths
            ]
            samples.append(
                {
                    "question": row["question"],
                    "paths": parsed_paths,
                    "path_scores": path_scores,
                    "answer": row.get("gold_answers", row.get("answers", [])),
                    "qid": row.get("qid", row.get("question_id", row["question"])),
                }
            )
    print(f"Loaded grouped scored paths: {len(samples)} samples from {path}")
    return samples


def load_raw_data(data_dir, split="train"):
    if os.path.isfile(data_dir) and data_dir.endswith(".jsonl"):
        return load_grouped_scored_paths(data_dir)

    file_path = os.path.join(data_dir, f"{split}_with_paths.json")
    with open(file_path, "r", encoding="utf-8") as f:
        raw_data = json.load(f)

    samples = []
    for item in raw_data:
        samples.append(
            {
                "question": item["question"],
                "paths": item["paths"],
                "path_scores": item["path_scores"],
                "answer": item["answer"],
            }
        )
    print(f"Loaded {split} split: {len(samples)} samples")
    return samples


def precompute_bert_embeddings(
    samples,
    bert_model,
    bert_tokenizer,
    max_paths=30,
    max_path_len=7,
    device="cuda",
    batch_size=256,
    save_path=None,
):
    bert_model = bert_model.to(device)
    bert_model.eval()

    all_texts = set()
    for sample in samples:
        all_texts.add(sample["question"])
        for path in sample["paths"][:max_paths]:
            for node in path[:max_path_len]:
                all_texts.add(node)

    all_texts = list(all_texts)
    print(f"Encoding {len(all_texts)} unique text entries")

    text_to_embed = {}
    for i in tqdm(range(0, len(all_texts), batch_size), desc="BERT encoding"):
        batch_texts = all_texts[i : i + batch_size]
        encoded = bert_tokenizer(
            batch_texts,
            padding=True,
            truncation=True,
            max_length=64,
            return_tensors="pt",
        ).to(device)
        with torch.no_grad():
            outputs = bert_model(**encoded)
            cls_embeds = outputs.last_hidden_state[:, 0, :].cpu()
        for text, embed in zip(batch_texts, cls_embeds):
            text_to_embed[text] = embed

    processed = []
    bert_dim = 768

    for sample in tqdm(samples, desc="Building samples"):
        question_embed = text_to_embed[sample["question"]]
        paths = sample["paths"][:max_paths]
        scores = sample["path_scores"][:max_paths]

        path_node_embeds = torch.zeros(max_paths, max_path_len, bert_dim)
        path_lengths = torch.zeros(max_paths, dtype=torch.long)
        path_scores = torch.zeros(max_paths)
        path_mask = torch.zeros(max_paths, dtype=torch.bool)

        for p_idx, path in enumerate(paths):
            path_len = min(len(path), max_path_len)
            path_lengths[p_idx] = path_len
            path_scores[p_idx] = scores[p_idx]
            path_mask[p_idx] = True
            for n_idx in range(path_len):
                path_node_embeds[p_idx, n_idx] = text_to_embed[path[n_idx]]

        processed.append(
            {
                "question_embed": question_embed,
                "path_node_embeds": path_node_embeds,
                "path_lengths": path_lengths,
                "path_scores": path_scores,
                "path_mask": path_mask,
                "answer": sample["answer"],
                "question": sample["question"],
                "paths_text": format_paths_text(paths),
            }
        )

    if save_path:
        torch.save(processed, save_path)
        print(f"Saved precomputed embeddings to {save_path}")

    return processed


class PrecomputedDataset(Dataset):
    def __init__(self, processed_samples):
        self.samples = processed_samples

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx]


def collate_fn(batch):
    return {
        "question_embed": torch.stack([b["question_embed"] for b in batch]),
        "path_node_embeds": torch.stack([b["path_node_embeds"] for b in batch]),
        "path_lengths": torch.stack([b["path_lengths"] for b in batch]),
        "path_scores": torch.stack([b["path_scores"] for b in batch]),
        "path_mask": torch.stack([b["path_mask"] for b in batch]),
        "answer": [b["answer"] for b in batch],
        "question": [b["question"] for b in batch],
        "paths_text": [b["paths_text"] for b in batch],
    }
