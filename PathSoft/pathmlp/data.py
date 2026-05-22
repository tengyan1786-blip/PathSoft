from __future__ import annotations

import json
import re
from collections import OrderedDict
from pathlib import Path

import torch
from torch.utils.data import Dataset


SHARD_RE = re.compile(r"shard_(\d+)_(\d+)\.pth$")


def shard_files(encoded_dir: Path) -> list[Path]:
    return sorted(path for path in encoded_dir.glob("shard_*.pth") if path.is_file())


class EncodedPathDataset(Dataset):
    def __init__(self, encoded_dir: str | Path, cache_size: int = 8):
        self.encoded_dir = Path(encoded_dir)
        self.cache_size = max(1, cache_size)
        self.shards = []
        self._cache: OrderedDict[int, list[dict]] = OrderedDict()

        files = shard_files(self.encoded_dir)
        if not files:
            raise FileNotFoundError(f"No encoded path items found in {self.encoded_dir}")
        manifest = self.encoded_dir / "manifest.json"
        self.num_records = None
        if manifest.is_file():
            with manifest.open("r", encoding="utf-8") as f:
                meta = json.load(f)
            self.num_records = int(meta["num_records"])

        for shard_file in files:
            match = SHARD_RE.match(shard_file.name)
            if match is None:
                raise ValueError(f"Unexpected shard filename: {shard_file}")
            start = int(match.group(1))
            end = int(match.group(2))
            self.shards.append((start, end, shard_file))

        if self.num_records is None:
            self.num_records = self.shards[-1][1]

    def __len__(self) -> int:
        return self.num_records

    @property
    def num_shards(self) -> int:
        return len(self.shards)

    def get_shard_size(self, shard_idx: int) -> int:
        start, end, _ = self.shards[shard_idx]
        return end - start

    def __getitem__(self, index: int) -> dict:
        if index < 0 or index >= self.num_records:
            raise IndexError(index)
        shard_idx = self._find_shard(index)
        items = self._load_shard(shard_idx)
        start, _, _ = self.shards[shard_idx]
        return items[index - start]

    def _find_shard(self, index: int) -> int:
        lo = 0
        hi = len(self.shards) - 1
        while lo <= hi:
            mid = (lo + hi) // 2
            start, end, _ = self.shards[mid]
            if index < start:
                hi = mid - 1
            elif index >= end:
                lo = mid + 1
            else:
                return mid
        raise IndexError(index)

    def _load_shard(self, shard_idx: int) -> list[dict]:
        if shard_idx in self._cache:
            items = self._cache.pop(shard_idx)
            self._cache[shard_idx] = items
            return items
        _, _, shard_file = self.shards[shard_idx]
        payload = torch.load(shard_file, map_location="cpu")
        items = payload["items"]
        self._cache[shard_idx] = items
        while len(self._cache) > self.cache_size:
            self._cache.popitem(last=False)
        return items


def collate_path_batch(rows: list[dict]) -> dict:
    return {
        "qid": [row["qid"] for row in rows],
        "question": [row["question"] for row in rows],
        "path_text": [row["path_text"] for row in rows],
        "path_edges": [row["path_edges"] for row in rows],
        "answers": [row["answers"] for row in rows],
        "answer_entity_ids": [row["answer_entity_ids"] for row in rows],
        "end_entity": [row["end_entity"] for row in rows],
        "end_entity_id": [row["end_entity_id"] for row in rows],
        "label": torch.tensor([row["label"] for row in rows], dtype=torch.float32),
        "q_emb": torch.stack([row["q_emb"] for row in rows], dim=0),
        "path_emb": torch.stack([row["path_emb"] for row in rows], dim=0),
    }
