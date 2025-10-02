# train_bert.py
"""
Minimal BERT-like model in PyTorch (didactic).
This version includes a full training and evaluation loop.
"""
import math
from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from transformers import BertTokenizer, get_linear_schedule_with_warmup
from datasets import load_dataset
import evaluate
from tqdm.auto import tqdm # For progress bars

# ==============================================================================
# --------------------------
# Output dataclass
# --------------------------
@dataclass
class SequenceClassifierOutput:
    loss: Optional[torch.Tensor] = None
    logits: Optional[torch.Tensor] = None
    hidden_states: Optional[torch.Tensor] = None
    pooler_output: Optional[torch.Tensor] = None

# --------------------------
# Utilities
# --------------------------
def gelu(x):
    return 0.5 * x * (1.0 + torch.erf(x / math.sqrt(2.0)))

# --------------------------
# Embeddings
# --------------------------
class BertEmbeddings(nn.Module):
    def __init__(self, vocab_size, hidden_size=256, max_position_embeddings=512, type_vocab_size=2, dropout=0.1):
        super().__init__()
        self.word_embeddings = nn.Embedding(vocab_size, hidden_size)
        self.position_embeddings = nn.Embedding(max_position_embeddings, hidden_size)
        self.token_type_embeddings = nn.Embedding(type_vocab_size, hidden_size)
        self.LayerNorm = nn.LayerNorm(hidden_size, eps=1e-12)
        self.dropout = nn.Dropout(dropout)

    def forward(self, input_ids, token_type_ids=None):
        seq_len = input_ids.size(1)
        if token_type_ids is None:
            token_type_ids = torch.zeros_like(input_ids)

        position_ids = torch.arange(seq_len, dtype=torch.long, device=input_ids.device).unsqueeze(0).expand_as(input_ids)
        w = self.word_embeddings(input_ids)
        p = self.position_embeddings(position_ids)
        t = self.token_type_embeddings(token_type_ids)
        embeddings = w + p + t
        embeddings = self.LayerNorm(embeddings)
        embeddings = self.dropout(embeddings)
        return embeddings

# --------------------------
# Multi-head Self-attention
# --------------------------
class MultiHeadSelfAttention(nn.Module):
    def __init__(self, hidden_size=256, num_attention_heads=8, dropout=0.1):
        super().__init__()
        assert hidden_size % num_attention_heads == 0
        self.num_heads = num_attention_heads
        self.head_dim = hidden_size // num_attention_heads
        self.scale = self.head_dim ** -0.5
        self.q = nn.Linear(hidden_size, hidden_size)
        self.k = nn.Linear(hidden_size, hidden_size)
        self.v = nn.Linear(hidden_size, hidden_size)
        self.out = nn.Linear(hidden_size, hidden_size)
        self.attn_dropout = nn.Dropout(dropout)
        self.proj_dropout = nn.Dropout(dropout)

    def transpose_for_scores(self, x):
        new_shape = x.size()[:-1] + (self.num_heads, self.head_dim)
        x = x.view(*new_shape)
        return x.permute(0, 2, 1, 3)

    def forward(self, hidden_states, attention_mask=None):
        q = self.transpose_for_scores(self.q(hidden_states))
        k = self.transpose_for_scores(self.k(hidden_states))
        v = self.transpose_for_scores(self.v(hidden_states))
        attn_scores = torch.matmul(q, k.transpose(-1, -2)) * self.scale
        if attention_mask is not None:
            attn_scores = attn_scores + attention_mask
        attn_probs = F.softmax(attn_scores, dim=-1)
        attn_probs = self.attn_dropout(attn_probs)
        context = torch.matmul(attn_probs, v)
        context = context.permute(0, 2, 1, 3).contiguous()
        new_context_shape = context.size()[:-2] + (self.num_heads * self.head_dim,)
        context = context.view(*new_context_shape)
        out = self.out(context)
        out = self.proj_dropout(out)
        return out

# --------------------------
# Feed-forward
# --------------------------
class FeedForward(nn.Module):
    def __init__(self, hidden_size=256, intermediate_size=1024, dropout=0.1):
        super().__init__()
        self.dense_1 = nn.Linear(hidden_size, intermediate_size)
        self.dense_2 = nn.Linear(intermediate_size, hidden_size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        x = self.dense_1(x)
        x = gelu(x)
        x = self.dense_2(x)
        x = self.dropout(x)
        return x

# --------------------------
# Transformer Encoder Layer
# --------------------------
class TransformerEncoderLayer(nn.Module):
    def __init__(self, hidden_size=256, num_attention_heads=8, intermediate_size=1024, dropout=0.1):
        super().__init__()
        self.attention = MultiHeadSelfAttention(hidden_size, num_attention_heads, dropout)
        self.attn_layer_norm = nn.LayerNorm(hidden_size, eps=1e-12)
        self.ffn = FeedForward(hidden_size, intermediate_size, dropout)
        self.ffn_layer_norm = nn.LayerNorm(hidden_size, eps=1e-12)

    def forward(self, x, attention_mask=None):
        attn_out = self.attention(x, attention_mask=attention_mask)
        x = self.attn_layer_norm(x + attn_out)
        ffn_out = self.ffn(x)
        x = self.ffn_layer_norm(x + ffn_out)
        return x

# --------------------------
# Transformer Encoder (stack)
# --------------------------
class TransformerEncoder(nn.Module):
    def __init__(self, num_hidden_layers=6, **layer_kwargs):
        super().__init__()
        self.layers = nn.ModuleList([TransformerEncoderLayer(**layer_kwargs) for _ in range(num_hidden_layers)])

    def forward(self, x, attention_mask=None):
        for layer in self.layers:
            x = layer(x, attention_mask=attention_mask)
        return x

# --------------------------
# BertModel
# --------------------------
class BertModel(nn.Module):
    def __init__(self, vocab_size, hidden_size=256, num_hidden_layers=4, num_attention_heads=8,
                 intermediate_size=1024, max_position_embeddings=512, type_vocab_size=2, dropout=0.1):
        super().__init__()
        self.embeddings = BertEmbeddings(vocab_size, hidden_size, max_position_embeddings, type_vocab_size, dropout)
        self.encoder = TransformerEncoder(num_hidden_layers=num_hidden_layers, hidden_size=hidden_size,
                                          num_attention_heads=num_attention_heads, intermediate_size=intermediate_size,
                                          dropout=dropout)
        self.pooler = nn.Linear(hidden_size, hidden_size)
        self.pooler_activation = nn.Tanh()

    def forward(self, input_ids, token_type_ids=None, attention_mask=None):
        embeddings = self.embeddings(input_ids, token_type_ids=token_type_ids)
        if attention_mask is not None:
            extended_mask = attention_mask.unsqueeze(1).unsqueeze(2).to(dtype=embeddings.dtype)
            extended_mask = (1.0 - extended_mask) * -10000.0
        else:
            extended_mask = None
        encoder_output = self.encoder(embeddings, attention_mask=extended_mask)
        pooled = self.pooler(encoder_output[:, 0])
        pooled = self.pooler_activation(pooled)
        return encoder_output, pooled

# --------------------------
# BertForSequenceClassification (wrapper)
# --------------------------
class BertForSequenceClassification(nn.Module):
    def __init__(self, bert: BertModel, num_labels=2, dropout=0.1):
        super().__init__()
        self.bert = bert
        hidden_size = self.bert.embeddings.word_embeddings.embedding_dim
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(hidden_size, num_labels)

    def forward(self, input_ids, token_type_ids=None, attention_mask=None, labels=None):
        encoder_output, pooled_output = self.bert(input_ids, token_type_ids=token_type_ids, attention_mask=attention_mask)
        pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output)
        loss = None
        if labels is not None:
            loss = F.cross_entropy(logits, labels)
        return SequenceClassifierOutput(loss=loss, logits=logits, hidden_states=encoder_output, pooler_output=pooled_output)

# ----------------------------------------------------
# Main Training and Evaluation Script
# ----------------------------------------------------
def main():
    # --- 1. Hyperparameters and Setup ---
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    NUM_EPOCHS = 3
    BATCH_SIZE = 16
    LEARNING_RATE = 5e-5
    MAX_SEQ_LENGTH = 128 # Max sequence length for tokenizer

    # --- 2. Load Dataset and Tokenizer ---
    # We use 'bert-base-uncased' tokenizer to match standard BERT practices
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

    # Load the SST-2 dataset from the GLUE benchmark
    raw_datasets = load_dataset("glue", "sst2")
    print("\nDataset loaded:")
    print(raw_datasets)
    print("\nExample from training set:", raw_datasets['train'][0])

    # --- 3. Preprocess Data ---
    def tokenize_function(examples):
        # The tokenizer handles padding and truncation
        return tokenizer(examples["sentence"], padding="max_length", truncation=True, max_length=MAX_SEQ_LENGTH)

    tokenized_datasets = raw_datasets.map(tokenize_function, batched=True)

    # Remove original text column and set format to PyTorch tensors
    tokenized_datasets = tokenized_datasets.remove_columns(["sentence", "idx"])
    tokenized_datasets = tokenized_datasets.rename_column("label", "labels")
    tokenized_datasets.set_format("torch")

    train_dataloader = DataLoader(tokenized_datasets["train"], shuffle=True, batch_size=BATCH_SIZE)
    eval_dataloader = DataLoader(tokenized_datasets["validation"], batch_size=BATCH_SIZE)

    # --- 4. Initialize Model, Optimizer, and Scheduler ---
    model = BertForSequenceClassification(
        bert=BertModel(
            vocab_size=tokenizer.vocab_size, # Use vocab size from the pre-trained tokenizer
            hidden_size=256,                 # Smaller hidden size for faster training
            num_hidden_layers=4,             # Fewer layers
            num_attention_heads=8,
            intermediate_size=1024,
            max_position_embeddings=MAX_SEQ_LENGTH
        ),
        num_labels=2 # SST-2 has 2 labels (positive/negative)
    ).to(device)

    print(f"\nModel initialized with {sum(p.numel() for p in model.parameters()):,} parameters.")

    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE)
    num_training_steps = NUM_EPOCHS * len(train_dataloader)
    lr_scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=0, num_training_steps=num_training_steps
    )

    # --- 5. Training and Evaluation Loop ---
    accuracy_metric = evaluate.load("accuracy")
    progress_bar = tqdm(range(num_training_steps))

    for epoch in range(NUM_EPOCHS):
        # --- Training ---
        model.train()
        for batch in train_dataloader:
            batch = {k: v.to(device) for k, v in batch.items()}
            outputs = model(**batch)
            loss = outputs.loss

            loss.backward()
            optimizer.step()
            lr_scheduler.step()
            optimizer.zero_grad()
            progress_bar.update(1)

        # --- Evaluation ---
        model.eval()
        for batch in eval_dataloader:
            batch = {k: v.to(device) for k, v in batch.items()}
            with torch.no_grad():
                outputs = model(**batch)

            logits = outputs.logits
            predictions = torch.argmax(logits, dim=-1)
            accuracy_metric.add_batch(predictions=predictions, references=batch["labels"])

        eval_metric = accuracy_metric.compute()
        print(f"\n--- Epoch {epoch + 1}/{NUM_EPOCHS} ---")
        print(f"Validation Accuracy: {eval_metric['accuracy']:.4f}")

if __name__ == "__main__":
    main()