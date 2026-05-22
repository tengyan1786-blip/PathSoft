from __future__ import annotations

import argparse
import heapq
import json
from itertools import count
from pathlib import Path

import torch
from tqdm import tqdm

from data import collate_path_batch, shard_files
from model import PathScorer


def iter_batches(rows: list[dict], batch_size: int):
    for start in range(0, len(rows), batch_size):
        yield rows[start : start + batch_size]


@torch.no_grad()
def main() -> None:
    parser = argparse.ArgumentParser('Stream-score candidate paths with a trained PathScorer.')
    parser.add_argument('--encoded-dir', required=True, type=Path)
    parser.add_argument('--checkpoint', required=True, type=Path)
    parser.add_argument('--output', required=True, type=Path)
    parser.add_argument('--top-k', type=int, default=200)
    parser.add_argument('--batch-size', type=int, default=512)
    args = parser.parse_args()

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    ckpt = torch.load(args.checkpoint, map_location='cpu')
    emb_dim = int(ckpt['config']['emb_dim'])
    model = PathScorer(emb_dim=emb_dim).to(device)
    model.load_state_dict(ckpt['model_state_dict'])
    model.eval()

    heaps: dict[str, list[tuple[float, int, dict]]] = {}
    metadata: dict[str, dict] = {}
    tie_breaker = count()

    files = shard_files(args.encoded_dir)
    if not files:
        raise FileNotFoundError(f'No encoded shards found in {args.encoded_dir}')

    for shard_file in tqdm(files, desc='score shards'):
        payload = torch.load(shard_file, map_location='cpu')
        items = payload['items']
        for rows in iter_batches(items, args.batch_size):
            batch = collate_path_batch(rows)
            scores = model(batch['q_emb'].to(device), batch['path_emb'].to(device)).cpu().tolist()
            for idx, score in enumerate(scores):
                qid = batch['qid'][idx]
                metadata.setdefault(
                    qid,
                    {
                        'qid': qid,
                        'question': batch['question'][idx],
                        'gold_answers': batch['answers'][idx],
                    },
                )
                record = {
                    'path_text': batch['path_text'][idx],
                    'path_edges': batch['path_edges'][idx],
                    'end_entity': batch['end_entity'][idx],
                    'end_entity_id': batch['end_entity_id'][idx],
                    'label': int(batch['label'][idx].item()),
                    'score': float(score),
                }
                heap = heaps.setdefault(qid, [])
                entry = (float(score), next(tie_breaker), record)
                if len(heap) < args.top_k:
                    heapq.heappush(heap, entry)
                elif score > heap[0][0]:
                    heapq.heapreplace(heap, entry)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open('w') as f:
        for qid in sorted(metadata):
            scored_paths = [entry[2] for entry in sorted(heaps.get(qid, []), reverse=True)]
            row = {**metadata[qid], 'scored_paths': scored_paths[: args.top_k]}
            f.write(json.dumps(row, ensure_ascii=False) + '\n')
    print(f'wrote {args.output}')


if __name__ == '__main__':
    main()
