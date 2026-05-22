import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import BertModel


class PathEncoder(nn.Module):
    def __init__(self, input_dim=768, hidden_dim=256, num_layers=2, dropout=0.1):
        super().__init__()
        self.gru = nn.GRU(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.compress = nn.Linear(hidden_dim * 2, hidden_dim)
        self.layer_norm = nn.LayerNorm(hidden_dim)

    def forward(self, path_embeddings, path_lengths):
        batch_size, num_paths, max_len, input_dim = path_embeddings.shape
        x = path_embeddings.view(batch_size * num_paths, max_len, input_dim)
        lengths = path_lengths.view(batch_size * num_paths).clamp(min=1)
        packed = nn.utils.rnn.pack_padded_sequence(
            x, lengths.cpu(), batch_first=True, enforce_sorted=False
        )
        _, hidden = self.gru(packed)
        forward_hidden = hidden[-2]
        backward_hidden = hidden[-1]
        combined = torch.cat([forward_hidden, backward_hidden], dim=-1)
        path_vectors = self.layer_norm(self.compress(combined))
        return path_vectors.view(batch_size, num_paths, -1)


class ScoreGuidedAttention(nn.Module):
    def __init__(self, hidden_dim=256, num_heads=4, use_score_prior=True):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = hidden_dim // num_heads
        self.use_score_prior = use_score_prior
        assert hidden_dim % num_heads == 0
        self.q_proj = nn.Linear(hidden_dim, hidden_dim)
        self.k_proj = nn.Linear(hidden_dim, hidden_dim)
        self.v_proj = nn.Linear(hidden_dim, hidden_dim)
        self.out_proj = nn.Linear(hidden_dim, hidden_dim)
        self.score_lambda = nn.Parameter(torch.ones(num_heads))
        self.layer_norm = nn.LayerNorm(hidden_dim)

    def forward(self, question_vec, path_vectors, path_scores, path_mask=None):
        batch_size, num_paths, _ = path_vectors.shape
        Q = self.q_proj(question_vec).unsqueeze(1)
        K = self.k_proj(path_vectors)
        V = self.v_proj(path_vectors)
        Q = Q.view(batch_size, 1, self.num_heads, self.head_dim).transpose(1, 2)
        K = K.view(batch_size, num_paths, self.num_heads, self.head_dim).transpose(1, 2)
        V = V.view(batch_size, num_paths, self.num_heads, self.head_dim).transpose(1, 2)
        attn_logits = torch.matmul(Q, K.transpose(-2, -1)) / (self.head_dim ** 0.5)
        if self.use_score_prior:
            score_bias = torch.log(path_scores.clamp(min=1e-6)).unsqueeze(1).unsqueeze(1)
            lambda_weight = self.score_lambda.view(1, self.num_heads, 1, 1)
            attn_logits = attn_logits + lambda_weight * score_bias
        if path_mask is not None:
            mask = ~path_mask.unsqueeze(1).unsqueeze(1)
            attn_logits = attn_logits.masked_fill(mask, float("-inf"))
        attn_weights = F.softmax(attn_logits, dim=-1).nan_to_num(0.0)
        out = torch.matmul(attn_weights, V)
        out = out.transpose(1, 2).contiguous().view(batch_size, -1)
        out = self.layer_norm(self.out_proj(out))
        return out, attn_weights.squeeze(2)


class Projector(nn.Module):
    def __init__(self, hidden_dim=256, llm_dim=4096, n_tokens=16, dropout=0.1):
        super().__init__()
        self.up_proj = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 4),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.token_heads = nn.ModuleList(
            [nn.Linear(hidden_dim * 4, llm_dim) for _ in range(n_tokens)]
        )

    def forward(self, aggregated):
        shared = self.up_proj(aggregated)
        tokens = [head(shared) for head in self.token_heads]
        return torch.stack(tokens, dim=1)


class PathSoftAdapter(nn.Module):
    def __init__(
        self,
        bert_model_name="bert-base-uncased",
        bert_dim=768,
        hidden_dim=256,
        gru_layers=2,
        num_heads=4,
        llm_dim=4096,
        n_tokens=16,
        dropout=0.1,
        freeze_bert=True,
        load_bert_backbone=False,
        use_score_prior=True,
    ):
        super().__init__()
        self.bert = None
        if load_bert_backbone:
            self.bert = BertModel.from_pretrained(bert_model_name)
            if freeze_bert:
                for param in self.bert.parameters():
                    param.requires_grad = False
            bert_dim = self.bert.config.hidden_size

        self.question_proj = nn.Linear(bert_dim, hidden_dim)
        self.path_encoder = PathEncoder(
            input_dim=bert_dim,
            hidden_dim=hidden_dim,
            num_layers=gru_layers,
            dropout=dropout,
        )
        self.attention = ScoreGuidedAttention(
            hidden_dim=hidden_dim,
            num_heads=num_heads,
            use_score_prior=use_score_prior,
        )
        self.projector = Projector(
            hidden_dim=hidden_dim,
            llm_dim=llm_dim,
            n_tokens=n_tokens,
            dropout=dropout,
        )

    def encode_texts_with_bert(self, input_ids, attention_mask):
        if self.bert is None:
            raise RuntimeError("BERT backbone is not loaded; use precomputed embeddings instead.")
        with torch.no_grad() if not any(p.requires_grad for p in self.bert.parameters()) else torch.enable_grad():
            outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        return outputs.last_hidden_state[:, 0, :]

    def forward(self, path_node_embeds, path_lengths, path_scores, question_embed, path_mask=None):
        question_vec = self.question_proj(question_embed)
        path_vectors = self.path_encoder(path_node_embeds, path_lengths)
        aggregated, attn_weights = self.attention(question_vec, path_vectors, path_scores, path_mask)
        soft_prompt = self.projector(aggregated)
        return soft_prompt, attn_weights


def count_trainable_params(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
