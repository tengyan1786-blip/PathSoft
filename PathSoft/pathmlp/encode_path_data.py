from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import torch
import torch.nn.functional as F
from tqdm import tqdm
from transformers import AutoModel, AutoTokenizer


class FrozenGTEEncoder:
    def __init__(self, model_path: str, device: torch.device, max_length: int):
        self.device = device
        self.max_length = max_length
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.model = AutoModel.from_pretrained(
            model_path,
            trust_remote_code=True,
            unpad_inputs=True,
            use_memory_efficient_attention=True,
        ).to(device)
        for param in self.model.parameters():
            param.requires_grad = False
        self.model.eval()

    @torch.no_grad()
    def encode(self, texts: list[str], batch_size: int) -> torch.Tensor:
        chunks = []
        for start in range(0, len(texts), batch_size):
            batch_texts = texts[start : start + batch_size]
            batch = self.tokenizer(
                batch_texts,
                max_length=self.max_length,
                padding=True,
                truncation=True,
                return_tensors="pt",
            ).to(self.device)
            outputs = self.model(**batch).last_hidden_state[:, 0]
            outputs = F.normalize(outputs, p=2, dim=1)
            chunks.append(outputs.cpu())
        return torch.cat(chunks, dim=0) if chunks else torch.zeros(0, 1024)


def read_jsonl(path: Path) -> list[dict]:
    rows = []
    with path.open() as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def save_shard(rows: list[dict], q_embs: torch.Tensor, path_embs: torch.Tensor, out_file: Path):
    items = []
    for idx, row in enumerate(rows):
        items.append(
            {
                "qid": row["qid"],
                "question": row["question"],
                "path_text": row["path_text"],
                "path_edges": row["path_edges"],
                "answers": row["answers"],
                "answer_entity_ids": row["answer_entity_ids"],
                "end_entity": row["end_entity"],
                "end_entity_id": row["end_entity_id"],
                "label": int(row["label"]),
                "q_emb": q_embs[idx],
                "path_emb": path_embs[idx],
            }
        )
    tmp_file = out_file.with_suffix(out_file.suffix + ".tmp")
    torch.save({"items": items}, tmp_file)
    os.replace(tmp_file, out_file)


def main() -> None:
    parser = argparse.ArgumentParser("Encode path-level data with a frozen text encoder.")
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--model-path", required=True, help="Local path or HuggingFace model name for the frozen text encoder.")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--shard-size", type=int, default=2048)
    parser.add_argument("--max-length", type=int, default=128)
    parser.add_argument("--limit-records", type=int, default=None)
    args = parser.parse_args()

    rows = read_jsonl(args.input)
    if args.limit_records is not None:
        rows = rows[: args.limit_records]

    args.output_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    encoder = FrozenGTEEncoder(args.model_path, device, args.max_length)

    manifest = {
        "input": str(args.input),
        "model_path": args.model_path,
        "max_length": args.max_length,
        "num_records": len(rows),
        "shard_size": args.shard_size,
    }

    for start in tqdm(range(0, len(rows), args.shard_size), desc="encode shards"):
        end = min(start + args.shard_size, len(rows))
        out_file = args.output_dir / f"shard_{start:08d}_{end:08d}.pth"
        if out_file.exists():
            continue
        shard_rows = rows[start:end]
        q_embs = encoder.encode([row["question"] for row in shard_rows], args.batch_size)
        path_embs = encoder.encode([row["path_text"] for row in shard_rows], args.batch_size)
        save_shard(shard_rows, q_embs, path_embs, out_file)

    with (args.output_dir / "manifest.json").open("w") as f:
        json.dump(manifest, f, indent=2)

    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
