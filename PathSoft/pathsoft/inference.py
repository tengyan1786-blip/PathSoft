import argparse
import json
import os
import random

import torch
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
from models.path_adapter import PathSoftAdapter
from train import build_cache_paths, build_hybrid_context


def prepare_eval_samples(samples, max_paths, path_selection, path_selection_seed, path_score_mode):
    if path_selection == 'topk' and path_score_mode == 'original':
        return samples

    prepared = []
    for idx, sample in enumerate(samples):
        paths = list(sample['paths'])
        scores = list(sample['path_scores'])
        if path_selection == 'random':
            rng = random.Random(path_selection_seed + idx)
            indices = list(range(len(paths)))
            rng.shuffle(indices)
            indices = indices[: min(max_paths, len(indices))]
            selected_paths = [paths[i] for i in indices]
            selected_scores = [scores[i] for i in indices]
        else:
            selected_paths = paths[:max_paths]
            selected_scores = scores[:max_paths]
        if path_score_mode == 'uniform':
            selected_scores = [1.0] * len(selected_scores)
        prepared.append(
            {
                'question': sample['question'],
                'paths': selected_paths,
                'path_scores': selected_scores,
                'answer': sample['answer'],
                'qid': sample.get('qid', sample['question']),
            }
        )
    return prepared


def build_eval_cache_path(base_eval_cache, max_paths, max_path_len, path_selection, path_selection_seed, path_score_mode):
    stem, ext = os.path.splitext(base_eval_cache)
    suffix = f'__{path_selection}_k{max_paths}_len{max_path_len}_{path_score_mode}'
    if path_selection == 'random':
        suffix += f'_seed{path_selection_seed}'
    return f'{stem}{suffix}{ext}'


@torch.no_grad()
def generate_with_soft_prompt(llm, llm_tokenizer, llm_input_embed, soft_prompt, context_texts, device_llm, max_context_tokens, max_new_tokens):
    ctx_encoded = llm_tokenizer(
        context_texts,
        padding=True,
        truncation=True,
        max_length=max_context_tokens,
        return_tensors='pt',
    )
    ctx_embeds = llm_input_embed(ctx_encoded['input_ids'].to(device_llm))
    soft_prompt = soft_prompt.to(device_llm, dtype=ctx_embeds.dtype)
    inputs_embeds = torch.cat([soft_prompt, ctx_embeds], dim=1)
    soft_attn = torch.ones(inputs_embeds.shape[0], soft_prompt.shape[1], dtype=torch.long).to(device_llm)
    attention_mask = torch.cat([soft_attn, ctx_encoded['attention_mask'].to(device_llm)], dim=1)
    outputs = llm.generate(
        inputs_embeds=inputs_embeds,
        attention_mask=attention_mask,
        max_new_tokens=max_new_tokens,
        do_sample=False,
        num_beams=1,
        pad_token_id=llm_tokenizer.pad_token_id,
    )
    return [llm_tokenizer.decode(output_ids, skip_special_tokens=True).strip() for output_ids in outputs]


@torch.no_grad()
def run_pathsoft(adapter, llm, llm_tokenizer, llm_input_embed, dataloader, device, device_llm, max_context_tokens, max_new_tokens):
    adapter.eval()
    raw_predictions = []
    all_answers = []
    all_questions = []
    all_qids = []
    for batch in tqdm(dataloader, desc='pathsoft infer'):
        soft_prompt, _ = adapter(
            path_node_embeds=batch['path_node_embeds'].to(device),
            path_lengths=batch['path_lengths'].to(device),
            path_scores=batch['path_scores'].to(device),
            question_embed=batch['question_embed'].to(device),
            path_mask=batch['path_mask'].to(device),
        )
        context_texts = [build_hybrid_context(paths_text, q) for paths_text, q in zip(batch['paths_text'], batch['question'])]
        raw_predictions.extend(
            generate_with_soft_prompt(
                llm, llm_tokenizer, llm_input_embed, soft_prompt,
                context_texts, device_llm, max_context_tokens, max_new_tokens,
            )
        )
        all_answers.extend(batch['answer'])
        all_questions.extend(batch['question'])
        all_qids.extend(batch.get('qid', [''] * len(batch['question'])) if isinstance(batch, dict) else [''] * len(batch['question']))
    return raw_predictions, all_answers, all_questions, all_qids


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--eval_path_file', type=str, required=True)
    parser.add_argument('--checkpoint', type=str, required=True)
    parser.add_argument('--cache_dir', type=str, default='./cache')
    parser.add_argument('--output_dir', type=str, default='./inference_results')
    parser.add_argument('--llm_model', type=str, default=None)
    parser.add_argument('--bert_model', type=str, default='bert-base-uncased')
    parser.add_argument('--llm_dtype', choices=['float16', 'bfloat16', 'float32'], default='float16')
    parser.add_argument('--llm_device_map', type=str, default='auto')
    parser.add_argument('--adapter_device', type=str, default='cuda:0')
    parser.add_argument('--batch_size', type=int, default=2)
    parser.add_argument('--max_paths', type=int, default=20)
    parser.add_argument('--max_path_len', type=int, default=7)
    parser.add_argument('--max_context_tokens', type=int, default=1024)
    parser.add_argument('--max_new_tokens', type=int, default=64)
    parser.add_argument('--path_selection', choices=['topk', 'random'], default='topk')
    parser.add_argument('--path_selection_seed', type=int, default=0)
    parser.add_argument('--path_score_mode', choices=['original', 'uniform'], default='original')
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    ckpt = torch.load(args.checkpoint, map_location='cpu')
    ckpt_args = ckpt['args']
    llm_model = args.llm_model or ckpt_args['llm_model']
    torch_dtype = {
        'float16': torch.float16,
        'bfloat16': torch.bfloat16,
        'float32': torch.float32,
    }[args.llm_dtype]

    llm_tokenizer = AutoTokenizer.from_pretrained(llm_model, trust_remote_code=True)
    if llm_tokenizer.pad_token is None:
        llm_tokenizer.pad_token = llm_tokenizer.eos_token
    llm = AutoModelForCausalLM.from_pretrained(
        llm_model,
        device_map=args.llm_device_map,
        torch_dtype=torch_dtype,
        low_cpu_mem_usage=True,
        trust_remote_code=True,
    )
    llm.eval()
    llm_input_embed = llm.get_input_embeddings()
    device_llm = llm_input_embed.weight.device
    device = torch.device(args.adapter_device)

    adapter = PathSoftAdapter(
        bert_model_name=args.bert_model,
        hidden_dim=ckpt_args.get('hidden_dim', 256),
        gru_layers=ckpt_args.get('gru_layers', 2),
        num_heads=ckpt_args.get('num_heads', 4),
        llm_dim=llm.config.hidden_size,
        n_tokens=ckpt_args.get('n_tokens', 16),
        dropout=ckpt_args.get('dropout', 0.1),
        freeze_bert=True,
        load_bert_backbone=False,
        use_score_prior=not ckpt_args.get('disable_score_prior', False),
    ).to(device)
    adapter.load_state_dict(ckpt['adapter_state_dict'])
    adapter.eval()

    _, base_eval_cache = build_cache_paths(args.cache_dir, ckpt_args['train_path_file'], args.eval_path_file)
    eval_cache = build_eval_cache_path(
        base_eval_cache, args.max_paths, args.max_path_len,
        args.path_selection, args.path_selection_seed, args.path_score_mode,
    )
    eval_samples = prepare_eval_samples(
        load_raw_data(args.eval_path_file, 'eval'),
        args.max_paths, args.path_selection, args.path_selection_seed, args.path_score_mode,
    )
    if not os.path.exists(eval_cache):
        bert_model = BertModel.from_pretrained(args.bert_model)
        bert_tokenizer = BertTokenizer.from_pretrained(args.bert_model)
        eval_processed = precompute_bert_embeddings(
            eval_samples, bert_model, bert_tokenizer,
            max_paths=args.max_paths, max_path_len=args.max_path_len, save_path=eval_cache,
        )
        del bert_model
    else:
        eval_processed = torch.load(eval_cache)

    eval_loader = DataLoader(
        PrecomputedDataset(eval_processed), batch_size=args.batch_size,
        shuffle=False, collate_fn=collate_fn, num_workers=4, pin_memory=True,
    )
    predictions, answers, questions, _ = run_pathsoft(
        adapter, llm, llm_tokenizer, llm_input_embed, eval_loader,
        device, device_llm, args.max_context_tokens, args.max_new_tokens,
    )

    output_path = os.path.join(args.output_dir, 'pathsoft_predictions.jsonl')
    with open(output_path, 'w', encoding='utf-8') as fout:
        for sample, question, answer, prediction in zip(eval_samples, questions, answers, predictions):
            fout.write(
                json.dumps(
                    {
                        'qid': sample.get('qid', question),
                        'question': question,
                        'gold_answers': answer,
                        'raw_prediction': prediction,
                    },
                    ensure_ascii=False,
                )
                + '\n'
            )
    print(f'Predictions saved to {output_path}')


if __name__ == '__main__':
    main()
