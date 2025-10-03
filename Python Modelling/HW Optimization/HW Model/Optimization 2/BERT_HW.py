# BERT_HW_GCU_SCU.py
"""
Trainable, hardware-aware BERT model that simulates the proposed FPGA architecture.

This version is updated to include:
1.  Approximate LayerNorm using Newton-Raphson for SQRT.
2.  Tiled dataflow simulation for the Systolic Array (32x32).
3.  Per-channel weight quantization.
4.  SCU (Softmax Compute Unit) and GCU (GELU Compute Unit).
5.  Shared EU (Exponential Unit) and DU (Division Unit).
6.  Quantization after SCU, GCU.
7.  The residual ADDITION operation is performed on quantized tensors.
8.  Q and K tensors are quantized before the attention score multiplication.
9.  The V tensor is quantized before the final attention multiplication.
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
from tqdm.auto import tqdm
import numpy as np

# ==============================================================================
# --------------------------
# Hardware-Aware Components (from SCU_GCU.py)
# --------------------------

class ExponentialUnit(nn.Module):
    """
    Exponential Unit (EU) - Shared between SCU and GCU
    Hardware-friendly base-2 exponential computation using piecewise linear approximation.
    Based on the paper's equation (10): 2^x = 2^frac(x) << int(x)
    """
    def __init__(self, num_segments=8):
        super().__init__()
        self.num_segments = num_segments
        coeffs = []
        for i in range(num_segments):
            x_start, x_end = i / num_segments, (i + 1) / num_segments
            y_start, y_end = 2 ** x_start, 2 ** x_end
            K = (y_end - y_start) / (x_end - x_start)
            B = y_start - K * x_start
            coeffs.append([K, B])
        self.register_buffer('coefficients', torch.tensor(coeffs, dtype=torch.float32))

    def forward(self, x, use_log2e_scaling=True):
        if use_log2e_scaling:
            log2e = 1.0 + 0.5 - 0.0625  # Hardware-friendly approximation of log2(e)
            x = x * log2e
        
        x_int = torch.floor(x).long()
        x_frac = torch.clamp(x - x_int.float(), 0, 0.999999)
        segment_idx = torch.clamp(torch.floor(x_frac * self.num_segments).long(), 0, self.num_segments - 1)
        
        K = self.coefficients[segment_idx, 0]
        B = self.coefficients[segment_idx, 1]
        frac_result = K * x_frac + B
        
        x_int_clamped = torch.clamp(x_int, -15, 15)
        return frac_result * (2.0 ** x_int_clamped.float())

class DivisionUnit(nn.Module):
    """
    Division Unit (DU) - Shared between SCU and GCU
    Hardware-friendly division using equations (11-12) from the paper.
    """
    def __init__(self):
        super().__init__()
        self.eu = ExponentialUnit()

    def leading_one_detector(self, x):
        abs_x = torch.clamp(torch.abs(x), min=1e-8)
        log2_x = torch.log2(abs_x)
        w = torch.floor(log2_x).long()
        m = (log2_x - w.float()) + 1.0
        return w, m

    def forward(self, numerator, denominator, add_one_to_denominator=False):
        if add_one_to_denominator:
            denominator = 1.0 + denominator
        
        w1, m1 = self.leading_one_detector(numerator)
        w2, m2 = self.leading_one_detector(denominator)
        
        exponent = (m1 + w1.float()) - (m2 + w2.float())
        result = self.eu(exponent, use_log2e_scaling=False)
        
        return result * (torch.sign(numerator) * torch.sign(denominator))

class FindMaxUnit(nn.Module):
    """ Find Max Unit (FMU) for SCU using parallel reduction. """
    def forward(self, x):
        max_vals, _ = torch.max(x, dim=-1, keepdim=True)
        return max_vals

class AdderTree(nn.Module):
    """ Adder Tree for SCU for parallel summation. """
    def forward(self, x):
        return torch.sum(x, dim=-1, keepdim=True)

class PolynomialUnit(nn.Module):
    """ Polynomial computation unit for GCU using hardware-friendly approximations. """
    def __init__(self):
        super().__init__()
        self.cubic_coeff = 0.03125 + 0.03125 # Approx 0.044715 with shifts
        self.s_x_coeff = -10.0 - 0.25 - 0.0625 # Approx -2*log2(e)*sqrt(2/pi) with shifts

    def forward(self, x):
        # h(x) is implicitly calculated within s(x)
        h_x_times_neg_2_log2e = (math.sqrt(2.0 / math.pi) * x) + (self.cubic_coeff * (x ** 3))
        s_x = self.s_x_coeff * h_x_times_neg_2_log2e
        return s_x

class SCU(nn.Module):
    """
    Softmax Compute Unit (SCU) - Hardware-friendly softmax implementation.
    """
    def __init__(self):
        super().__init__()
        self.fmu = FindMaxUnit()
        self.eu = ExponentialUnit()
        self.adder_tree = AdderTree()
        self.du = DivisionUnit()

    def forward(self, scores):
        x_max = self.fmu(scores)
        shifted_scores = scores - x_max
        exp_scores = self.eu(shifted_scores, use_log2e_scaling=True)
        sum_exp = self.adder_tree(exp_scores)
        probabilities = self.du(exp_scores, sum_exp, add_one_to_denominator=False)
        # Final re-normalization for stability
        stable_probs = probabilities / (torch.sum(probabilities, dim=-1, keepdim=True) + 1e-9)
        return Quantize.apply(stable_probs)

class GCU(nn.Module):
    """
    GELU Compute Unit (GCU) - Hardware-friendly GELU implementation.
    """
    def __init__(self):
        super().__init__()
        self.polynomial_unit = PolynomialUnit()
        self.eu = ExponentialUnit()
        self.du = DivisionUnit()

    def forward(self, x):
        # Stage 1: Polynomial computation to get s(x) = -2*log2(e)*h(x)
        s_x = self.polynomial_unit(x)
        # Stage 2: Exponential computation of 2^s(x)
        exp_term = self.eu(s_x, use_log2e_scaling=False)
        # Stage 3: Division x / (1 + exp_term)
        return self.du(x, exp_term, add_one_to_denominator=True)

# --------------------------
# Original Hardware-Aware Components
# --------------------------

class Quantize(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, num_bits=8):
        qmax = 2**(num_bits - 1) - 1
        if x.numel() == 0: return x
        scale = x.abs().max() / qmax if x.dim() > 2 else x.abs().max(dim=1, keepdim=True)[0] / qmax
        scale[scale == 0] = 1.0
        q = torch.clamp((x / scale).round(), -qmax, qmax)
        return q * scale

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output, None

class QuantizedLinear(nn.Module):
    def __init__(self, in_features, out_features, bias=True):
        super().__init__()
        self.in_features, self.out_features = in_features, out_features
        self.weight = nn.Parameter(torch.Tensor(out_features, in_features))
        if bias: self.bias = nn.Parameter(torch.Tensor(out_features))
        else: self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in); nn.init.uniform_(self.bias, -bound, bound)

    def forward(self, input):
        q_input = Quantize.apply(input)
        q_weight = Quantize.apply(self.weight)
        return F.linear(q_input, q_weight, self.bias)

class ApproximateLayerNorm(nn.Module):
    def __init__(self, normalized_shape, eps=1e-5, nr_iterations=8):
        super().__init__()
        self.normalized_shape = normalized_shape
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.nr_iterations = nr_iterations

    def _sqrt_newton_raphson(self, S):
        x = torch.where(S > 1.0, S * 0.5, torch.ones_like(S))
        for _ in range(self.nr_iterations):
            x = 0.5 * (x + S / (x + 1e-9))
        return x

    def forward(self, x):
        mean = x.mean(dim=-1, keepdim=True)
        var = x.var(dim=-1, keepdim=True, unbiased=False)
        std_approx = self._sqrt_newton_raphson(var)
        x_norm = (x - mean) / (std_approx + self.eps)
        return self.weight * x_norm + self.bias

# ==============================================================================
# --------------------------
# BERT Model with Integrated SCU/GCU
# --------------------------

@dataclass
class SequenceClassifierOutput:
    loss: Optional[torch.Tensor] = None
    logits: Optional[torch.Tensor] = None

class BertEmbeddings(nn.Module):
    def __init__(self, vocab_size, hidden_size=256, max_position_embeddings=512, type_vocab_size=2, dropout=0.1):
        super().__init__()
        self.word_embeddings = nn.Embedding(vocab_size, hidden_size)
        self.position_embeddings = nn.Embedding(max_position_embeddings, hidden_size)
        self.token_type_embeddings = nn.Embedding(type_vocab_size, hidden_size)
        self.LayerNorm = ApproximateLayerNorm(hidden_size, eps=1e-12)
        self.dropout = nn.Dropout(dropout)

    def forward(self, input_ids, token_type_ids=None):
        seq_len = input_ids.size(1)
        if token_type_ids is None: token_type_ids = torch.zeros_like(input_ids)
        pos_ids = torch.arange(seq_len, dtype=torch.long, device=input_ids.device).unsqueeze(0).expand_as(input_ids)
        embeddings = self.word_embeddings(input_ids) + self.position_embeddings(pos_ids) + self.token_type_embeddings(token_type_ids)
        return self.dropout(self.LayerNorm(embeddings))

class MultiHeadSelfAttention(nn.Module):
    def __init__(self, hidden_size=256, num_attention_heads=8, dropout=0.1):
        super().__init__()
        self.num_heads, self.head_dim = num_attention_heads, hidden_size // num_attention_heads
        self.scale = self.head_dim ** -0.5
        self.q, self.k, self.v = QuantizedLinear(hidden_size, hidden_size), QuantizedLinear(hidden_size, hidden_size), QuantizedLinear(hidden_size, hidden_size)
        self.out = QuantizedLinear(hidden_size, hidden_size)
        self.softmax = SCU()  # <-- INTEGRATED SCU
        self.attn_dropout = nn.Dropout(dropout)
        self.proj_dropout = nn.Dropout(dropout)

    def transpose_for_scores(self, x):
        return x.view(x.size()[:-1] + (self.num_heads, self.head_dim)).permute(0, 2, 1, 3)

    def forward(self, hidden_states, attention_mask=None):
        TILE_SIZE = 32
        q_full, k_full, v_full = self.transpose_for_scores(self.q(hidden_states)), self.transpose_for_scores(self.k(hidden_states)), self.transpose_for_scores(self.v(hidden_states))

        attn_scores = torch.zeros((q_full.size(0), q_full.size(1), q_full.size(2), k_full.size(2)), device=q_full.device)
        for i in range(0, q_full.size(2), TILE_SIZE):
            for j in range(0, k_full.size(2), TILE_SIZE):
                # Quantize Q and K tiles before multiplication
                q_tile_quantized = Quantize.apply(q_full[:, :, i:i+TILE_SIZE, :])
                k_tile_quantized = Quantize.apply(k_full[:, :, j:j+TILE_SIZE, :])
                
                attn_scores[:, :, i:i+TILE_SIZE, j:j+TILE_SIZE] = torch.matmul(q_tile_quantized, k_tile_quantized.transpose(-1, -2)) * self.scale
        
        if attention_mask is not None: attn_scores = attn_scores + attention_mask
        attn_probs = self.attn_dropout(self.softmax(attn_scores))
        
        # MODIFICATION: Quantize the V tensor before the final multiplication
        v_full_quantized = Quantize.apply(v_full)
        
        context = torch.zeros_like(v_full)
        for i in range(0, attn_probs.size(2), TILE_SIZE):
            probs_tile = attn_probs[:, :, i:i+TILE_SIZE, :] # attn_probs is already quantized from SCU output
            context[:, :, i:i+TILE_SIZE, :] = torch.matmul(probs_tile, v_full_quantized)

        context = context.permute(0, 2, 1, 3).contiguous().view(hidden_states.size())
        return self.proj_dropout(self.out(context))

class FeedForward(nn.Module):
    def __init__(self, hidden_size=256, intermediate_size=1024, dropout=0.1):
        super().__init__()
        self.dense_1 = QuantizedLinear(hidden_size, intermediate_size)
        self.dense_2 = QuantizedLinear(intermediate_size, hidden_size)
        self.activation = GCU() # <-- INTEGRATED GCU
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        TILE_SIZE = 32
        out1 = torch.zeros(x.size(0), x.size(1), self.dense_1.out_features, device=x.device)
        for i in range(0, x.size(1), TILE_SIZE):
            out1[:, i:i+TILE_SIZE, :] = self.dense_1(x[:, i:i+TILE_SIZE, :])
        
        activated = Quantize.apply(self.activation(out1))
        
        out2 = torch.zeros_like(x)
        for i in range(0, activated.size(1), TILE_SIZE):
             out2[:, i:i+TILE_SIZE, :] = self.dense_2(activated[:, i:i+TILE_SIZE, :])
        return self.dropout(out2)

class TransformerEncoderLayer(nn.Module):
    def __init__(self, hidden_size=256, num_attention_heads=8, intermediate_size=1024, dropout=0.1):
        super().__init__()
        self.attention = MultiHeadSelfAttention(hidden_size, num_attention_heads, dropout)
        self.attn_layer_norm = ApproximateLayerNorm(hidden_size, eps=1e-12)
        self.ffn = FeedForward(hidden_size, intermediate_size, dropout)
        self.ffn_layer_norm = ApproximateLayerNorm(hidden_size, eps=1e-12)

    def forward(self, x, attention_mask=None):
        attn_output = self.attention(x, attention_mask=attention_mask)
        # Quantize each tensor BEFORE the addition
        norm_input_1 = Quantize.apply(x) + Quantize.apply(attn_output)
        x = self.attn_layer_norm(norm_input_1)
        
        ffn_output = self.ffn(x)
        # Quantize each tensor BEFORE the addition
        norm_input_2 = Quantize.apply(x) + Quantize.apply(ffn_output)
        x = self.ffn_layer_norm(norm_input_2)
        
        return x

class TransformerEncoder(nn.Module):
    def __init__(self, num_hidden_layers=4, **layer_kwargs):
        super().__init__()
        self.layers = nn.ModuleList([TransformerEncoderLayer(**layer_kwargs) for _ in range(num_hidden_layers)])

    def forward(self, x, attention_mask=None):
        for layer in self.layers: x = layer(x, attention_mask=attention_mask)
        return x

class BertModel(nn.Module):
    def __init__(self, vocab_size, hidden_size=256, num_hidden_layers=4, num_attention_heads=8,
                 intermediate_size=1024, max_position_embeddings=512, type_vocab_size=2, dropout=0.1):
        super().__init__()
        self.embeddings = BertEmbeddings(vocab_size, hidden_size, max_position_embeddings, type_vocab_size, dropout)
        self.encoder = TransformerEncoder(num_hidden_layers=num_hidden_layers, hidden_size=hidden_size,
                                          num_attention_heads=num_attention_heads, intermediate_size=intermediate_size, dropout=dropout)
        self.pooler = nn.Linear(hidden_size, hidden_size)
        self.pooler_activation = nn.Tanh()

    def forward(self, input_ids, token_type_ids=None, attention_mask=None):
        embeddings = self.embeddings(input_ids, token_type_ids=token_type_ids)
        extended_mask = (1.0 - attention_mask.unsqueeze(1).unsqueeze(2)) * -10000.0 if attention_mask is not None else None
        encoder_output = self.encoder(embeddings, attention_mask=extended_mask)
        pooled = self.pooler_activation(self.pooler(encoder_output[:, 0]))
        return encoder_output, pooled

class BertForSequenceClassification(nn.Module):
    def __init__(self, bert: BertModel, num_labels=2, dropout=0.1):
        super().__init__()
        self.bert = bert
        self.dropout = nn.Dropout(dropout)
        self.classifier = QuantizedLinear(bert.pooler.in_features, num_labels)

    def forward(self, input_ids, token_type_ids=None, attention_mask=None, labels=None):
        _, pooled_output = self.bert(input_ids, token_type_ids=token_type_ids, attention_mask=attention_mask)
        logits = self.classifier(self.dropout(pooled_output))
        loss = F.cross_entropy(logits.view(-1, self.classifier.out_features), labels.view(-1)) if labels is not None else None
        return SequenceClassifierOutput(loss=loss, logits=logits)

# ==============================================================================
# --------------------------
# Main Training and Evaluation Script
# --------------------------
def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    NUM_EPOCHS, BATCH_SIZE, LEARNING_RATE, MAX_SEQ_LENGTH = 3, 16, 5e-5, 128

    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    raw_datasets = load_dataset("glue", "sst2")

    def tokenize_function(examples):
        return tokenizer(examples["sentence"], padding="max_length", truncation=True, max_length=MAX_SEQ_LENGTH)

    tokenized_datasets = raw_datasets.map(tokenize_function, batched=True).remove_columns(["sentence", "idx"]).rename_column("label", "labels")
    tokenized_datasets.set_format("torch")

    train_dataloader = DataLoader(tokenized_datasets["train"], shuffle=True, batch_size=BATCH_SIZE)
    eval_dataloader = DataLoader(tokenized_datasets["validation"], batch_size=BATCH_SIZE)

    model = BertForSequenceClassification(
        bert=BertModel(
            vocab_size=tokenizer.vocab_size, hidden_size=256, num_hidden_layers=4,
            num_attention_heads=8, intermediate_size=1024, max_position_embeddings=MAX_SEQ_LENGTH
        ),
        num_labels=2
    ).to(device)

    print(f"\nAdvanced Hardware-Aware Model with SCU/GCU initialized with {sum(p.numel() for p in model.parameters()):,} parameters.")
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE)
    num_training_steps = NUM_EPOCHS * len(train_dataloader)
    lr_scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=int(0.1 * num_training_steps), num_training_steps=num_training_steps)
    
    accuracy_metric = evaluate.load("accuracy")
    progress_bar = tqdm(range(num_training_steps))

    for epoch in range(NUM_EPOCHS):
        model.train()
        for batch in train_dataloader:
            batch = {k: v.to(device) for k, v in batch.items()}
            loss = model(**batch).loss
            loss.backward()
            optimizer.step(); lr_scheduler.step(); optimizer.zero_grad()
            progress_bar.update(1)

        model.eval()
        for batch in eval_dataloader:
            batch = {k: v.to(device) for k, v in batch.items()}
            with torch.no_grad():
                logits = model(**batch).logits
            predictions = torch.argmax(logits, dim=-1)
            accuracy_metric.add_batch(predictions=predictions, references=batch["labels"])
            
        eval_metric = accuracy_metric.compute()
        print(f"\n--- Epoch {epoch + 1}/{NUM_EPOCHS} ---\nValidation Accuracy: {eval_metric['accuracy']:.4f}")

if __name__ == "__main__":
    main()