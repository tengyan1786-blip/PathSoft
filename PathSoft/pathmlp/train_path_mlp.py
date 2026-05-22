from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.optim import Adam
from torch.utils.data import DataLoader, Sampler
from tqdm import tqdm

from data import EncodedPathDataset, collate_path_batch
from model import PathScorer


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


class ShardShuffleSampler(Sampler[int]):
    """Shuffle shard order each epoch while keeping sequential reads within shards."""

    def __init__(self, dataset: EncodedPathDataset):
        self.dataset = dataset
        self.shard_ranges: list[range] = []
        start = 0
        for shard_idx in range(dataset.num_shards):
            shard_size = dataset.get_shard_size(shard_idx)
            self.shard_ranges.append(range(start, start + shard_size))
            start += shard_size

    def __iter__(self):
        shard_order = list(range(len(self.shard_ranges)))
        random.shuffle(shard_order)
        for shard_idx in shard_order:
            yield from self.shard_ranges[shard_idx]

    def __len__(self) -> int:
        return len(self.dataset)


def run_epoch(model, loader, optimizer, device, train: bool) -> float:
    model.train(mode=train)
    losses = []
    desc = 'train' if train else 'validation'
    context = torch.enable_grad() if train else torch.no_grad()
    with context:
        for batch in tqdm(loader, desc=desc, leave=False):
            q_emb = batch['q_emb'].to(device)
            path_emb = batch['path_emb'].to(device)
            labels = batch['label'].to(device)
            scores = model(q_emb, path_emb)
            loss = F.binary_cross_entropy(scores, labels)
            if train:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            losses.append(float(loss.item()))
    return sum(losses) / max(len(losses), 1)


def make_loader(dataset, batch_size, num_workers, cache_mode, shuffle_mode=None):
    device_is_cuda = torch.cuda.is_available()
    kwargs = {
        'batch_size': batch_size,
        'collate_fn': collate_path_batch,
        'num_workers': num_workers,
        'pin_memory': device_is_cuda,
    }
    if num_workers > 0:
        kwargs['persistent_workers'] = True
    if shuffle_mode == 'full':
        kwargs['shuffle'] = True
    elif shuffle_mode == 'shard':
        kwargs['sampler'] = ShardShuffleSampler(dataset)
    elif shuffle_mode == 'none':
        kwargs['shuffle'] = False
    return DataLoader(dataset, **kwargs)


def main() -> None:
    parser = argparse.ArgumentParser('Train PathMLP path scorer.')
    parser.add_argument('--train-dir', required=True, type=Path)
    parser.add_argument('--val-dir', required=True, type=Path)
    parser.add_argument('--output-dir', required=True, type=Path)
    parser.add_argument('--epochs', type=int, default=20)
    parser.add_argument('--patience', type=int, default=5)
    parser.add_argument('--batch-size', type=int, default=128)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--cache-size', type=int, default=4)
    parser.add_argument('--num-workers', type=int, default=4)
    parser.add_argument('--shuffle-mode', choices=['none', 'shard', 'full'], default='shard')
    args = parser.parse_args()

    set_seed(args.seed)
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    args.output_dir.mkdir(parents=True, exist_ok=True)

    train_set = EncodedPathDataset(args.train_dir, cache_size=args.cache_size)
    val_set = EncodedPathDataset(args.val_dir, cache_size=args.cache_size)
    train_loader = make_loader(train_set, args.batch_size, args.num_workers, args.cache_size, args.shuffle_mode)
    val_loader = make_loader(val_set, args.batch_size, args.num_workers, args.cache_size, None)

    emb_dim = int(train_set[0]['q_emb'].shape[-1])
    model = PathScorer(emb_dim=emb_dim).to(device)
    optimizer = Adam(model.parameters(), lr=args.lr)

    best_val_loss = float('inf')
    bad_epochs = 0
    history = []
    for epoch in range(args.epochs):
        train_loss = run_epoch(model, train_loader, optimizer, device, train=True)
        val_loss = run_epoch(model, val_loader, optimizer, device, train=False)
        row = {'epoch': epoch, 'train_loss': train_loss, 'val_loss': val_loss}
        history.append(row)
        print(json.dumps(row, indent=2))

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            bad_epochs = 0
            torch.save(
                {
                    'model_state_dict': model.state_dict(),
                    'config': {'emb_dim': emb_dim, 'lr': args.lr, 'batch_size': args.batch_size},
                    'state': {'best_val_loss': best_val_loss, 'epoch': epoch},
                },
                args.output_dir / 'best_model.pth',
            )
        else:
            bad_epochs += 1
        if bad_epochs >= args.patience:
            break

    with (args.output_dir / 'train_history.json').open('w') as f:
        json.dump(history, f, indent=2)


if __name__ == '__main__':
    main()
