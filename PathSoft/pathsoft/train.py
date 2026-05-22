import argparse
import json
import logging
import os
import random
from pathlib import Path

import torch
import torch.nn as nn
from torch.amp import GradScaler, autocast
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer, BertModel, BertTokenizer

import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data.data_pipeline import (
    PrecomputedDataset,
    collate_fn,
    load_raw_data,
    precompute_bert_embeddings,
)
from models.path_adapter import PathSoftAdapter, count_trainable_params

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def sanitize_name(value: str) -> str:
    return Path(value).stem.replace('.', '_').replace('/', '_')


def build_cache_paths(cache_dir: str, train_source: str, eval_source: str):
    train_name = sanitize_name(train_source)
    eval_name = sanitize_name(eval_source)
    return (
        os.path.join(cache_dir, f'train_{train_name}_bert_embeds.pt'),
        os.path.join(cache_dir, f'eval_{eval_name}_bert_embeds.pt'),
    )


def format_target_answers(answer) -> str:
    if isinstance(answer, list):
        answers = [str(item).strip() for item in answer if str(item).strip()]
    else:
        answers = [str(answer).strip()] if str(answer).strip() else []
    if not answers:
        return 'ans: not available'
    return '\n'.join(f'ans: {item}' for item in answers)


def build_hybrid_context(paths_text: str, question: str) -> str:
    question = question.strip()
    if not question.endswith('?'):
        question += '?'
    return (
        'Based on the knowledge graph paths, please answer the question. '
        'Please return formatted answers as a list, each prefixed with "ans:".\n\n'
        f'Paths:\n{paths_text}\n\n'
        f'Question:\n{question}\n\n'
        'Answer:'
    )


class PathSoftTrainer:
    def __init__(self, args):
        self.args = args
        self.device = torch.device(args.adapter_device)
        self._load_models()
        self._setup_optimizer()

    def _load_models(self):
        args = self.args
        logger.info('Loading frozen LLM...')
        self.llm_tokenizer = AutoTokenizer.from_pretrained(args.llm_model, trust_remote_code=True)
        if self.llm_tokenizer.pad_token is None:
            self.llm_tokenizer.pad_token = self.llm_tokenizer.eos_token

        torch_dtype = {
            'float16': torch.float16,
            'bfloat16': torch.bfloat16,
            'float32': torch.float32,
        }[args.llm_dtype]

        self.llm = AutoModelForCausalLM.from_pretrained(
            args.llm_model,
            device_map=args.llm_device_map,
            torch_dtype=torch_dtype,
            low_cpu_mem_usage=True,
            trust_remote_code=True,
        )
        self.llm.eval()
        for param in self.llm.parameters():
            param.requires_grad = False

        self.llm_input_embed = self.llm.get_input_embeddings()
        self.device_llm = self.llm_input_embed.weight.device
        llm_dim = self.llm.config.hidden_size
        logger.info('LLM hidden_size: %s', llm_dim)

        self.adapter = PathSoftAdapter(
            bert_model_name=args.bert_model,
            hidden_dim=args.hidden_dim,
            gru_layers=args.gru_layers,
            num_heads=args.num_heads,
            llm_dim=llm_dim,
            n_tokens=args.n_tokens,
            dropout=args.dropout,
            freeze_bert=True,
            load_bert_backbone=False,
            use_score_prior=not args.disable_score_prior,
        ).to(self.device)
        logger.info('Trainable adapter params: %s', f'{count_trainable_params(self.adapter):,}')

    def _setup_optimizer(self):
        self.optimizer = torch.optim.AdamW(
            filter(lambda p: p.requires_grad, self.adapter.parameters()),
            lr=self.args.lr,
            weight_decay=self.args.weight_decay,
        )
        self.scaler = GradScaler('cuda', enabled=self.args.llm_dtype != 'float32')

    def prepare_llm_input(self, soft_prompt, questions, answers, paths_texts):
        batch_size = len(questions)
        context_texts = [build_hybrid_context(paths_text, q) for paths_text, q in zip(paths_texts, questions)]
        answer_texts = [format_target_answers(a) for a in answers]

        ctx_encoded = self.llm_tokenizer(
            context_texts,
            padding=True,
            truncation=True,
            max_length=self.args.max_context_tokens,
            return_tensors='pt',
        )
        a_encoded = self.llm_tokenizer(
            answer_texts,
            padding=True,
            truncation=True,
            max_length=self.args.max_answer_tokens,
            return_tensors='pt',
            add_special_tokens=False,
        )

        with torch.no_grad():
            ctx_embeds = self.llm_input_embed(ctx_encoded['input_ids'].to(self.device_llm))
            a_embeds = self.llm_input_embed(a_encoded['input_ids'].to(self.device_llm))

        soft_prompt = soft_prompt.to(self.device_llm, dtype=ctx_embeds.dtype)
        inputs_embeds = torch.cat([soft_prompt, ctx_embeds, a_embeds], dim=1)

        n_tokens = soft_prompt.shape[1]
        ctx_len = ctx_embeds.shape[1]
        ignore_labels = torch.full((batch_size, n_tokens + ctx_len), -100, dtype=torch.long).to(self.device_llm)
        answer_labels = a_encoded['input_ids'].to(self.device_llm)
        answer_labels[answer_labels == self.llm_tokenizer.pad_token_id] = -100
        labels = torch.cat([ignore_labels, answer_labels], dim=1)

        soft_attn = torch.ones(batch_size, n_tokens, dtype=torch.long).to(self.device_llm)
        attention_mask = torch.cat(
            [soft_attn, ctx_encoded['attention_mask'].to(self.device_llm), a_encoded['attention_mask'].to(self.device_llm)],
            dim=1,
        )
        return inputs_embeds, labels, attention_mask

    def run_epoch(self, dataloader, epoch=None, train=True):
        self.adapter.train(mode=train)
        total_loss = 0.0
        num_batches = 0
        desc = f'Epoch {epoch}' if train else 'validation'
        pbar = tqdm(dataloader, desc=desc)
        context = torch.enable_grad() if train else torch.no_grad()
        with context:
            for batch in pbar:
                soft_prompt, _ = self.adapter(
                    path_node_embeds=batch['path_node_embeds'].to(self.device),
                    path_lengths=batch['path_lengths'].to(self.device),
                    path_scores=batch['path_scores'].to(self.device),
                    question_embed=batch['question_embed'].to(self.device),
                    path_mask=batch['path_mask'].to(self.device),
                )
                inputs_embeds, labels, attention_mask = self.prepare_llm_input(
                    soft_prompt,
                    batch['question'],
                    batch['answer'],
                    batch['paths_text'],
                )

                with autocast('cuda', enabled=self.args.llm_dtype != 'float32'):
                    outputs = self.llm(
                        inputs_embeds=inputs_embeds,
                        attention_mask=attention_mask,
                        labels=labels,
                    )
                    loss = outputs.loss

                if train:
                    self.optimizer.zero_grad()
                    self.scaler.scale(loss).backward()
                    self.scaler.unscale_(self.optimizer)
                    nn.utils.clip_grad_norm_(self.adapter.parameters(), max_norm=1.0)
                    self.scaler.step(self.optimizer)
                    self.scaler.update()

                total_loss += float(loss.item())
                num_batches += 1
                pbar.set_postfix({'loss': f'{loss.item():.4f}'})
        return total_loss / max(num_batches, 1)

    def save_checkpoint(self, path, epoch, eval_loss):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        torch.save(
            {
                'epoch': epoch,
                'adapter_state_dict': self.adapter.state_dict(),
                'optimizer_state_dict': self.optimizer.state_dict(),
                'state': {'eval_loss': eval_loss},
                'args': vars(self.args),
            },
            path,
        )
        logger.info('Checkpoint saved to %s', path)

    def load_checkpoint(self, path):
        ckpt = torch.load(path, map_location='cpu')
        self.adapter.load_state_dict(ckpt['adapter_state_dict'])
        if 'optimizer_state_dict' in ckpt:
            self.optimizer.load_state_dict(ckpt['optimizer_state_dict'])
        start_epoch = int(ckpt.get('epoch', 0)) + 1
        eval_loss = float(ckpt.get('state', {}).get('eval_loss', 'inf'))
        return start_epoch, eval_loss


def maybe_subset(dataset, subset_size, seed):
    if subset_size and subset_size > 0 and subset_size < len(dataset):
        rng = random.Random(seed)
        indices = rng.sample(range(len(dataset)), subset_size)
        return torch.utils.data.Subset(dataset, indices)
    return dataset


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--train_path_file', type=str, required=True)
    parser.add_argument('--eval_path_file', type=str, required=True)
    parser.add_argument('--cache_dir', type=str, default='./cache')
    parser.add_argument('--output_dir', type=str, default='./outputs')
    parser.add_argument('--max_paths', type=int, default=20)
    parser.add_argument('--max_path_len', type=int, default=7)
    parser.add_argument('--bert_model', type=str, default='bert-base-uncased')
    parser.add_argument('--llm_model', type=str, required=True)
    parser.add_argument('--llm_dtype', choices=['float16', 'bfloat16', 'float32'], default='float16')
    parser.add_argument('--llm_device_map', type=str, default='auto')
    parser.add_argument('--adapter_device', type=str, default='cuda:0')
    parser.add_argument('--hidden_dim', type=int, default=256)
    parser.add_argument('--gru_layers', type=int, default=2)
    parser.add_argument('--num_heads', type=int, default=4)
    parser.add_argument('--n_tokens', type=int, default=16)
    parser.add_argument('--dropout', type=float, default=0.1)
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--batch_size', type=int, default=2)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--weight_decay', type=float, default=0.01)
    parser.add_argument('--eval_every', type=int, default=1)
    parser.add_argument('--max_context_tokens', type=int, default=1024)
    parser.add_argument('--max_answer_tokens', type=int, default=64)
    parser.add_argument('--resume_from', type=str, default=None)
    parser.add_argument('--disable_score_prior', action='store_true')
    parser.add_argument('--train_subset_size', type=int, default=0)
    parser.add_argument('--train_subset_seed', type=int, default=0)
    parser.add_argument('--eval_subset_size', type=int, default=0)
    parser.add_argument('--eval_subset_seed', type=int, default=0)
    args = parser.parse_args()

    os.makedirs(args.cache_dir, exist_ok=True)
    os.makedirs(args.output_dir, exist_ok=True)

    train_cache, eval_cache = build_cache_paths(args.cache_dir, args.train_path_file, args.eval_path_file)
    if not os.path.exists(train_cache) or not os.path.exists(eval_cache):
        logger.info('Precomputing frozen BERT embeddings...')
        bert_model = BertModel.from_pretrained(args.bert_model)
        bert_tokenizer = BertTokenizer.from_pretrained(args.bert_model)
        train_samples = load_raw_data(args.train_path_file, 'train')
        train_processed = precompute_bert_embeddings(
            train_samples, bert_model, bert_tokenizer,
            max_paths=args.max_paths, max_path_len=args.max_path_len, save_path=train_cache,
        )
        eval_samples = load_raw_data(args.eval_path_file, 'val')
        eval_processed = precompute_bert_embeddings(
            eval_samples, bert_model, bert_tokenizer,
            max_paths=args.max_paths, max_path_len=args.max_path_len, save_path=eval_cache,
        )
        del bert_model
    else:
        logger.info('Loading cached BERT embeddings...')
        train_processed = torch.load(train_cache)
        eval_processed = torch.load(eval_cache)

    train_dataset = maybe_subset(PrecomputedDataset(train_processed), args.train_subset_size, args.train_subset_seed)
    eval_dataset = maybe_subset(PrecomputedDataset(eval_processed), args.eval_subset_size, args.eval_subset_seed)

    train_loader = DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True,
        collate_fn=collate_fn, num_workers=4, pin_memory=True,
    )
    eval_loader = DataLoader(
        eval_dataset, batch_size=args.batch_size, shuffle=False,
        collate_fn=collate_fn, num_workers=4, pin_memory=True,
    )

    trainer = PathSoftTrainer(args)
    best_eval_loss = float('inf')
    start_epoch = 1
    if args.resume_from:
        logger.info('Resuming from checkpoint: %s', args.resume_from)
        start_epoch, best_eval_loss = trainer.load_checkpoint(args.resume_from)

    history = []
    for epoch in range(start_epoch, args.epochs + 1):
        train_loss = trainer.run_epoch(train_loader, epoch=epoch, train=True)
        row = {'epoch': epoch, 'train_loss': train_loss}
        if epoch % args.eval_every == 0 or epoch == args.epochs:
            eval_loss = trainer.run_epoch(eval_loader, train=False)
            row['eval_loss'] = eval_loss
            if eval_loss < best_eval_loss:
                best_eval_loss = eval_loss
                trainer.save_checkpoint(os.path.join(args.output_dir, 'best_adapter.pt'), epoch, eval_loss)
        history.append(row)
        with open(os.path.join(args.output_dir, 'train_history.json'), 'w', encoding='utf-8') as fout:
            json.dump(history, fout, ensure_ascii=False, indent=2)
        logger.info('Epoch %s | train_loss=%.4f | best_eval_loss=%.4f', epoch, train_loss, best_eval_loss)

    logger.info('Training done. Best eval_loss: %.4f', best_eval_loss)


if __name__ == '__main__':
    main()
