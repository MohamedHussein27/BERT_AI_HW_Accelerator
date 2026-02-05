# BERT_HW_WITH_SCALE_EXTRACTION.py
"""
Modified BERT training script with scale extraction for hardware implementation.

NEW FEATURES (added without changing original structure):
1. Extracts input activation scales for each linear layer
2. Records GELU input ranges for hardware design
3. Saves all scales to .npz file with weights
4. Generates scale statistics report

USAGE: Just run this instead of your original script - everything else stays the same!
"""
import math
from dataclasses import dataclass
from typing import Optional
import os
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from transformers import BertTokenizer, get_linear_schedule_with_warmup
from datasets import load_dataset
import evaluate
from tqdm.auto import tqdm
import numpy as np
from collections import defaultdict

# ==============================================================================
# GLOBAL SCALE TRACKER
# ==============================================================================
class ScaleTracker:
    """
    Global tracker for activation scales and GELU input ranges.
    Collects statistics during training without affecting model behavior.
    """
    def __init__(self):
        self.layer_input_scales = defaultdict(list)  # input scales per layer
        self.gelu_input_ranges = []  # (min, max) for GELU inputs
        self.enabled = False  # Only collect during evaluation to avoid training overhead
    
    def record_input_scale(self, layer_name, scale):
        """Record input scale for a layer."""
        if self.enabled and len(self.layer_input_scales[layer_name]) < 500:  # Limit samples
            self.layer_input_scales[layer_name].append(scale)
    
    def record_gelu_input(self, x):
        """Record GELU input range."""
        if self.enabled:
            # Collect batch statistics
            if len(self.gelu_input_ranges) < 2000:
                min_val = x.min().item()
                max_val = x.max().item()
                mean_val = x.mean().item()
                std_val = x.std().item()
                self.gelu_input_ranges.append({
                    'min': min_val,
                    'max': max_val,
                    'mean': mean_val,
                    'std': std_val
                })
    
    def get_statistics(self):
        """Compute statistics from collected data."""
        stats = {}
        
        # Layer input scales
        for layer_name, scales in self.layer_input_scales.items():
            if scales:
                stats[layer_name] = {
                    'mean': float(np.mean(scales)),
                    'median': float(np.median(scales)),
                    'std': float(np.std(scales)),
                    'min': float(np.min(scales)),
                    'max': float(np.max(scales)),
                    'p95': float(np.percentile(scales, 95)),
                    'samples': len(scales)
                }
        
        # GELU input ranges
        if self.gelu_input_ranges:
            all_mins = [x['min'] for x in self.gelu_input_ranges]
            all_maxs = [x['max'] for x in self.gelu_input_ranges]
            all_means = [x['mean'] for x in self.gelu_input_ranges]
            all_stds = [x['std'] for x in self.gelu_input_ranges]
            
            stats['gelu_input_range'] = {
                'overall_min': float(np.min(all_mins)),
                'overall_max': float(np.max(all_maxs)),
                'typical_min_p05': float(np.percentile(all_mins, 5)),
                'typical_max_p95': float(np.percentile(all_maxs, 95)),
                'mean_of_means': float(np.mean(all_means)),
                'mean_of_stds': float(np.mean(all_stds)),
                'median_min': float(np.median(all_mins)),
                'median_max': float(np.median(all_maxs)),
                'samples': len(self.gelu_input_ranges)
            }
        
        return stats
    
    def enable(self):
        """Enable scale collection."""
        self.enabled = True
    
    def disable(self):
        """Disable scale collection."""
        self.enabled = False

# Global instance
scale_tracker = ScaleTracker()

# ==============================================================================
# --------------------------
# Hardware-Aware Components
# --------------------------

# Per-Channel/Tile Quantization 
class Quantize(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, num_bits=8):
        qmax = 2**(num_bits - 1) - 1
        if x.numel() == 0: return x
        if x.dim() > 1: max_val = x.abs().max(dim=-1, keepdim=True)[0]
        else: max_val = x.abs().max()
        scale = max_val / qmax
        scale[scale == 0] = 1.0
        q = torch.clamp((x / scale).round(), -qmax, qmax)
        return q * scale

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output, None

class QuantizedLinear(nn.Module):
    def __init__(self, in_features, out_features, bias=True, layer_name=''):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.layer_name = layer_name  # *** NEW: for tracking
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

        if scale_tracker.enabled:
            with torch.no_grad():
                abs_max = input.abs().max().item()
                input_scale = abs_max / 127.0 if abs_max > 0 else 1.0
                scale_tracker.record_input_scale(self.layer_name, input_scale)
        
        q_input = Quantize.apply(input)
        q_weight = Quantize.apply(self.weight)
        return F.linear(q_input, q_weight, self.bias)

# Fixed-Point Arithmetic Simulation 
def to_fixed_point(x, bits, frac_bits):
    scale = 2.0**frac_bits
    min_val, max_val = -(2.0**(bits - 1)), (2.0**(bits - 1)) - 1
    fixed = torch.clamp((x * scale).round(), min_val, max_val)
    return fixed / scale

class PLASoftmax(nn.Module):
    def __init__(self, num_intervals=12, domain_min=-10.0, domain_max=0.0):
        super().__init__()
        self.domain_min, self.domain_max = domain_min, domain_max
        coeffs_np = self._build_pla_coeffs(num_intervals, domain_min, domain_max)
        for key in ['coeffs_m', 'coeffs_c', 'intervals']:
            self.register_buffer(key, torch.tensor([c[{'coeffs_m':'m', 'coeffs_c':'c', 'intervals':'a'}[key]] for c in coeffs_np], dtype=torch.float32))

    @staticmethod
    def _build_pla_coeffs(num_intervals, domain_min, domain_max):
        xs, ys = np.linspace(domain_min, domain_max, 1001), np.exp(np.linspace(domain_min, domain_max, 1001))
        intervals = np.linspace(domain_min, domain_max, num_intervals + 1)
        coeffs = []
        for i in range(num_intervals):
            a, b = intervals[i], intervals[i+1]
            mask = (xs >= a) & (xs <= b)
            m, c = np.polyfit(xs[mask], ys[mask], 1)
            coeffs.append({'a': a, 'm': m, 'c': c})
        return coeffs

    def pla_exp(self, x):
        x_clamped = torch.clamp(x, self.domain_min, self.domain_max)
        indices = torch.clamp(torch.sum(x_clamped.unsqueeze(-1) >= self.intervals, dim=-1) - 1, 0, len(self.intervals) - 2)
        return self.coeffs_m[indices] * x_clamped + self.coeffs_c[indices]

    def forward(self, scores):
        max_scores, _ = scores.max(dim=-1, keepdim=True)
        shifted_scores = scores - max_scores
        shifted_scores_fx = to_fixed_point(shifted_scores, 32, 26)
        exps = self.pla_exp(shifted_scores_fx)
        softmax_output = exps / (exps.sum(dim=-1, keepdim=True) + 1e-9)
        return Quantize.apply(softmax_output)

# Newton Raphson for LayerNorm
class ApproximateLayerNorm(nn.Module):
    def __init__(self, normalized_shape, eps=1e-5, elementwise_affine=True, nr_iterations=8):
        super().__init__()
        self.normalized_shape = normalized_shape
        self.eps = eps
        self.elementwise_affine = elementwise_affine
        if self.elementwise_affine:
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
        var_fx = to_fixed_point(var, 32, 26)
        std_approx = self._sqrt_newton_raphson(var_fx)
        x = (x - mean) / (std_approx + self.eps)
        if self.elementwise_affine:
            x = self.weight * x + self.bias
        return x

# ==============================================================================
# --- GCU (GELU COMPUTE UNIT) 
# ==============================================================================
class ExponentialUnit(nn.Module):
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
            log2e = 1.0 + 0.5 - 0.0625
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

class PolynomialUnit(nn.Module):
    def __init__(self):
        super().__init__()
        self.sqrt_2_over_pi = 0.8
        self.cubic_coeff = 0.03125 + 0.03125
        self.s_x_coeff = -10.0 - 0.25 - 0.0625

    def forward(self, x):
        h_x = self.sqrt_2_over_pi * x + self.cubic_coeff * (x ** 3)
        return self.s_x_coeff * h_x

class GCU(nn.Module):
    def __init__(self):
        super().__init__()
        self.polynomial_unit = PolynomialUnit()
        self.eu = ExponentialUnit()
        self.du = DivisionUnit()

    def forward(self, x):
        #Record GELU input range for hardware design
        if scale_tracker.enabled:
            with torch.no_grad():
                scale_tracker.record_gelu_input(x)
        
        s_x = self.polynomial_unit(x)
        exp_term = self.eu(-s_x, use_log2e_scaling=False)
        return self.du(x, exp_term, add_one_to_denominator=True)

# --------------------------
# Output dataclass
# --------------------------
@dataclass
class SequenceClassifierOutput:
    loss: Optional[torch.Tensor] = None
    logits: Optional[torch.Tensor] = None

# ==============================================================================
# BERT Model with Hardware-Aware Layers
# ==============================================================================
class BertEmbeddings(nn.Module):
    def __init__(self, vocab_size, hidden_size=768, max_position_embeddings=512, type_vocab_size=2, dropout=0.1):
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
    def __init__(self, hidden_size=768, num_attention_heads=12, dropout=0.1, layer_id=0):
        super().__init__()
        self.num_heads, self.head_dim = num_attention_heads, hidden_size // num_attention_heads
        self.scale = self.head_dim ** -0.5
     
        self.q = QuantizedLinear(hidden_size, hidden_size, layer_name=f'bert.encoder.layers.{layer_id}.attention.q')
        self.k = QuantizedLinear(hidden_size, hidden_size, layer_name=f'bert.encoder.layers.{layer_id}.attention.k')
        self.v = QuantizedLinear(hidden_size, hidden_size, layer_name=f'bert.encoder.layers.{layer_id}.attention.v')
        self.out = QuantizedLinear(hidden_size, hidden_size, layer_name=f'bert.encoder.layers.{layer_id}.attention.out')
        self.softmax = PLASoftmax()
        self.attn_dropout = nn.Dropout(dropout)
        self.proj_dropout = nn.Dropout(dropout)

    def transpose_for_scores(self, x):
        return x.view(x.size()[:-1] + (self.num_heads, self.head_dim)).permute(0, 2, 1, 3)

    def forward(self, hidden_states, attention_mask=None):
        TILE_SIZE = 32
        q_full = self.transpose_for_scores(self.q(hidden_states))
        k_full = self.transpose_for_scores(self.k(hidden_states))
        v_full = self.transpose_for_scores(self.v(hidden_states))

        attn_scores = torch.zeros((q_full.size(0), q_full.size(1), q_full.size(2), k_full.size(2)), device=q_full.device)
        for i in range(0, q_full.size(2), TILE_SIZE):
            for j in range(0, k_full.size(2), TILE_SIZE):
                q_tile_quantized = Quantize.apply(q_full[:, :, i:i+TILE_SIZE, :])
                k_tile_quantized = Quantize.apply(k_full[:, :, j:j+TILE_SIZE, :])
                attn_scores[:, :, i:i+TILE_SIZE, j:j+TILE_SIZE] = torch.matmul(q_tile_quantized, k_tile_quantized.transpose(-1, -2)) * self.scale
        
        if attention_mask is not None: attn_scores = attn_scores + attention_mask
        attn_probs = self.attn_dropout(self.softmax(attn_scores))
        v_full_quantized = Quantize.apply(v_full)
        context = torch.zeros_like(v_full)
        for i in range(0, attn_probs.size(2), TILE_SIZE):
            probs_tile = attn_probs[:, :, i:i+TILE_SIZE, :]
            context[:, :, i:i+TILE_SIZE, :] = torch.matmul(probs_tile, v_full_quantized)

        context = context.permute(0, 2, 1, 3).contiguous().view(hidden_states.size())
        return self.proj_dropout(self.out(context))

class FeedForward(nn.Module):
    def __init__(self, hidden_size=768, intermediate_size=3072, dropout=0.1, layer_id=0):
        super().__init__()
      
        self.dense_1 = QuantizedLinear(hidden_size, intermediate_size, layer_name=f'bert.encoder.layers.{layer_id}.ffn.dense_1')
        self.dense_2 = QuantizedLinear(intermediate_size, hidden_size, layer_name=f'bert.encoder.layers.{layer_id}.ffn.dense_2')
        self.dropout = nn.Dropout(dropout)
        self.gcu = GCU()

    def forward(self, x):
        TILE_SIZE = 32
        out1 = torch.zeros(x.size(0), x.size(1), self.dense_1.out_features, device=x.device)
        for i in range(0, x.size(1), TILE_SIZE):
            out1[:, i:i+TILE_SIZE, :] = self.dense_1(x[:, i:i+TILE_SIZE, :])
        
        x = Quantize.apply(self.gcu(out1))
        
        out2 = torch.zeros(x.size(0), x.size(1), self.dense_2.out_features, device=x.device)
        for i in range(0, x.size(1), TILE_SIZE):
             out2[:, i:i+TILE_SIZE, :] = self.dense_2(x[:, i:i+TILE_SIZE, :])
        return self.dropout(out2)

class TransformerEncoderLayer(nn.Module):
    def __init__(self, hidden_size=768, num_attention_heads=12, intermediate_size=3072, dropout=0.1, layer_id=0):
        super().__init__()
      
        self.attention = MultiHeadSelfAttention(hidden_size, num_attention_heads, dropout, layer_id=layer_id)
        self.attn_layer_norm = ApproximateLayerNorm(hidden_size, eps=1e-12)
        self.ffn = FeedForward(hidden_size, intermediate_size, dropout, layer_id=layer_id)
        self.ffn_layer_norm = ApproximateLayerNorm(hidden_size, eps=1e-12)

    def forward(self, x, attention_mask=None):
        attn_output = self.attention(x, attention_mask=attention_mask)
        norm_input_1 = Quantize.apply(x) + Quantize.apply(attn_output)
        x = self.attn_layer_norm(norm_input_1)
        ffn_output = self.ffn(x)
        norm_input_2 = Quantize.apply(x) + Quantize.apply(ffn_output)
        x = self.ffn_layer_norm(norm_input_2)
        return x

class TransformerEncoder(nn.Module):
    def __init__(self, num_hidden_layers=12, **layer_kwargs):
        super().__init__()
        
        self.layers = nn.ModuleList([TransformerEncoderLayer(**layer_kwargs, layer_id=i) for i in range(num_hidden_layers)])

    def forward(self, x, attention_mask=None):
        for layer in self.layers:
            x = layer(x, attention_mask=attention_mask)
        return x

class BertModel(nn.Module):
    def __init__(self, vocab_size, hidden_size=768, num_hidden_layers=12, num_attention_heads=12,
                 intermediate_size=3072, max_position_embeddings=512, type_vocab_size=2, dropout=0.1):
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
        
        self.classifier = QuantizedLinear(bert.pooler.in_features, num_labels, layer_name='classifier')

    def forward(self, input_ids, token_type_ids=None, attention_mask=None, labels=None):
        _, pooled_output = self.bert(input_ids, token_type_ids=token_type_ids, attention_mask=attention_mask)
        logits = self.classifier(self.dropout(pooled_output))
        loss = F.cross_entropy(logits.view(-1, self.classifier.out_features), labels.view(-1)) if labels is not None else None
        return SequenceClassifierOutput(loss=loss, logits=logits)

# ==============================================================================
# Main Training and Evaluation Script
# ==============================================================================
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
            vocab_size=tokenizer.vocab_size, 
            hidden_size=768, 
            num_hidden_layers=12,
            num_attention_heads=12, 
            intermediate_size=3072, 
            max_position_embeddings=512
        ),
        num_labels=2
    ).to(device)

    print(f"\nBERT-base Hardware-Aware Model initialized with {sum(p.numel() for p in model.parameters()):,} parameters.")
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE)
    num_training_steps = NUM_EPOCHS * len(train_dataloader)
    lr_scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=int(0.1 * num_training_steps), num_training_steps=num_training_steps)
    
    accuracy_metric = evaluate.load("accuracy")
    progress_bar = tqdm(range(num_training_steps))

    for epoch in range(NUM_EPOCHS):
        model.train()
        scale_tracker.disable()  # Disable during training for speed
        
        for batch in train_dataloader:
            batch = {k: v.to(device) for k, v in batch.items()}
            loss = model(**batch).loss
            loss.backward()
            optimizer.step(); lr_scheduler.step(); optimizer.zero_grad()
            progress_bar.update(1)

        # Enable scale collection during evaluation 
        print(f"\n--- Epoch {epoch + 1}/{NUM_EPOCHS} ---")
        print("Collecting activation scales and GELU input ranges...")
        model.eval()
        scale_tracker.enable()
        
        for i, batch in enumerate(eval_dataloader):
            batch = {k: v.to(device) for k, v in batch.items()}
            with torch.no_grad():
                logits = model(**batch).logits
            predictions = torch.argmax(logits, dim=-1)
            accuracy_metric.add_batch(predictions=predictions, references=batch["labels"])
            
            # Collect scales from 100 batches
            if i >= 100:
                break
        
        scale_tracker.disable()  # Disable after collection
            
        eval_metric = accuracy_metric.compute()
        print(f"Validation Accuracy: {eval_metric['accuracy']:.4f}")

    print("\nTraining complete. Extracting weights and scales...")
    
    # Use an ABSOLUTE path to save the weights file.
    output_weights_file = 'D:/weights_with_scales.npz'
    extract_and_save_quantized_weights(model, output_weights_file)


# ==============================================================================
# Weight Extraction with Scale Information 
# ==============================================================================
def extract_and_save_quantized_weights(model: nn.Module, file_path: str):
    """
    Performs post-training quantization (per-tensor) on model parameters
    and saves them to a compressed NumPy file (.npz) WITH SCALE INFORMATION.
    """
    import json
    
    torch.cuda.empty_cache()
    quantized_state_dict = {}
    
    print("\n" + "="*70)
    print("EXTRACTING WEIGHTS AND SCALES FOR HARDWARE")
    print("="*70)
    
    # Get scale statistics
    scale_stats = scale_tracker.get_statistics()
    
    # Quantize weights and biases
    for name, param in model.named_parameters():
        if param.requires_grad:
            param_data = param.detach().cpu().numpy()
            
            # Quantize weights (2D matrices)
            if 'weight' in name and param_data.ndim == 2:
                abs_max = np.abs(param_data).max()
                weight_scale = abs_max / 127.0 if abs_max > 0 else 1.0
                
                quantized_param = np.round(param_data / weight_scale).clip(-128, 127).astype(np.int8)
                quantized_state_dict[name] = quantized_param
                quantized_state_dict[f"{name}.scale"] = np.float32(weight_scale)
                
                # *** NEW: Add measured input scale for this layer ***
                # Try to find matching layer in scale_stats
                # The layer_name should match the parameter name without ".weight"
                layer_key = name.replace('.weight', '')
                
                if layer_key in scale_stats and 'median' in scale_stats[layer_key]:
                    input_scale = scale_stats[layer_key]['median']
                    quantized_state_dict[f"{name}.input_scale"] = np.float32(input_scale)
                    print(f"  {name}: weight_scale={weight_scale:.6e}, input_scale={input_scale:.6e}")
                else:
                    # Fallback to 1.0
                    quantized_state_dict[f"{name}.input_scale"] = np.float32(1.0)
                    print(f"  {name}: weight_scale={weight_scale:.6e}, input_scale=1.0 (default)")
            
            # Quantize biases to INT32 with matched scale
            elif 'bias' in name and param_data.ndim == 1:
                weight_name = name.replace('.bias', '.weight')
                weight_scale_key = f"{weight_name}.scale"
                input_scale_key = f"{weight_name}.input_scale"
                
                if weight_scale_key in quantized_state_dict and input_scale_key in quantized_state_dict:
                    weight_scale = float(quantized_state_dict[weight_scale_key])
                    input_scale = float(quantized_state_dict[input_scale_key])
                    
                    # Calculate bias scale (matched to matmul output)
                    bias_scale = input_scale * weight_scale
                    
                    # Quantize to INT32
                    bias_int32 = np.round(param_data / bias_scale).clip(-2147483648, 2147483647).astype(np.int32)
                    
                    quantized_state_dict[name] = bias_int32
                    quantized_state_dict[f"{name}.scale"] = np.float32(bias_scale)
                    
                    print(f"  {name}: INT32, bias_scale={bias_scale:.6e}")
                else:
                    # Fallback: keep as FP32
                    quantized_state_dict[name] = param_data.astype(np.float32)
                    print(f"  {name}: FP32 (fallback)")
            
            # Other parameters (embeddings, LayerNorm, etc.)
            else:
                # Quantize if 2D, otherwise keep as FP32
                if param_data.ndim >= 2:
                    abs_max = np.abs(param_data).max()
                    scale = abs_max / 127.0 if abs_max > 0 else 1.0
                    quantized_param = np.round(param_data / scale).clip(-128, 127).astype(np.int8)
                    quantized_state_dict[name] = quantized_param
                    quantized_state_dict[f"{name}.scale"] = np.float32(scale)
                else:
                    quantized_state_dict[name] = param_data.astype(np.float32)

    # Save weights
    np.savez_compressed(file_path, **quantized_state_dict)
    print(f"\n✓ Saved quantized weights to {file_path}")
    
    # *** NEW: Save scale statistics as JSON ***
    metadata_file = file_path.replace('.npz', '_scale_metadata.json')
    with open(metadata_file, 'w') as f:
        json.dump(scale_stats, f, indent=2)
    print(f"✓ Saved scale metadata to {metadata_file}")
    
    # *** NEW: Print GELU input range (critical for hardware design) ***
    if 'gelu_input_range' in scale_stats:
        gelu_stats = scale_stats['gelu_input_range']
        print("\n" + "="*70)
        print("GELU INPUT RANGE ANALYSIS (CRITICAL FOR HARDWARE!)")
        print("="*70)
        
        print(f"\n MEASURED RANGES FROM {gelu_stats['samples']} BATCHES:")
        print(f"  Overall range:        [{gelu_stats['overall_min']:.4f}, {gelu_stats['overall_max']:.4f}]")
        print(f"  Typical range (90%):  [{gelu_stats['typical_min_p05']:.4f}, {gelu_stats['typical_max_p95']:.4f}]")
        print(f"  Median range:         [{gelu_stats['median_min']:.4f}, {gelu_stats['median_max']:.4f}]")
        print(f"  Mean of values:       {gelu_stats['mean_of_means']:.4f}")
        print(f"  Typical std dev:      {gelu_stats['mean_of_stds']:.4f}")
        
        # Calculate fixed-point format recommendation
        max_abs = max(abs(gelu_stats['overall_min']), abs(gelu_stats['overall_max']))
        int_bits = int(np.ceil(np.log2(max_abs + 1))) + 1  # +1 for sign
        frac_bits = 16 - int_bits
        
        print(f"\n🔧 HARDWARE DESIGN RECOMMENDATIONS:")
        print(f"\n  1. PRIMARY OPERATING RANGE (90% of inputs):")
        print(f"     [{gelu_stats['typical_min_p05']:.3f}, {gelu_stats['typical_max_p95']:.3f}]")
        print(f"     → Design your GCU with maximum precision here")
        
        print(f"\n  2. FULL RANGE TO SUPPORT:")
        print(f"     [{gelu_stats['overall_min']:.3f}, {gelu_stats['overall_max']:.3f}]")
        print(f"     → Handle with saturation/clipping")
        
        print(f"\n  3. RECOMMENDED FIXED-POINT FORMAT:")
        print(f"     Q{int_bits}.{frac_bits} (16-bit)")
        print(f"     • Integer bits:    {int_bits} (including sign)")
        print(f"     • Fractional bits: {frac_bits}")
        print(f"     • Range:           [{-2**(int_bits-1):.1f}, {2**(int_bits-1)-1:.1f}]")
        print(f"     • Precision:       {2**-frac_bits:.6f}")
        
        print(f"\n  4. LOOKUP TABLE (LUT) SIZE ESTIMATE:")
        range_span = gelu_stats['typical_max_p95'] - gelu_stats['typical_min_p05']
        lut_entries_001 = int(range_span / 0.01)
        lut_entries_0001 = int(range_span / 0.001)
        print(f"     • With 0.01 precision:  ~{lut_entries_001:,} entries  ({lut_entries_001*2:,} bytes)")
        print(f"     • With 0.001 precision: ~{lut_entries_0001:,} entries ({lut_entries_0001*2:,} bytes)")
        
        print(f"\n  5. PIECEWISE LINEAR APPROXIMATION:")
        for n_segments in [8, 16, 32]:
            segment_size = range_span / n_segments
            print(f"     • {n_segments:2d} segments: {segment_size:.4f} per segment ({n_segments*4} bytes for slopes/intercepts)")
        
        print("="*70)
    
    # Print summary
    print("\n" + "="*70)
    print("EXTRACTION SUMMARY")
    print("="*70)
    total_size = sum(p.nbytes for p in quantized_state_dict.values())
    print(f"Total size: {total_size / (1024*1024):.2f} MB")
    print(f"Layers with measured input scales: {len([k for k in quantized_state_dict.keys() if '.input_scale' in k])}")
    print(f"INT32 biases: {sum(1 for k, v in quantized_state_dict.items() if 'bias' in k and v.dtype == np.int32)}")
    print("="*70)

if __name__ == "__main__":
    main()