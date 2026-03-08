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

class ScaleTracker:
    """
    Tracks per-tensor activation scales and GELU input ranges during evaluation.
    Collected scales are saved into the .npz for hardware register programming.
    """
    def __init__(self):
        self.layer_input_scales = defaultdict(list)
        self.gelu_input_ranges  = []
        self.enabled = False

    def record_input_scale(self, layer_name, scale):
        if self.enabled and len(self.layer_input_scales[layer_name]) < 500:
            self.layer_input_scales[layer_name].append(scale)

    def record_gelu_input(self, x):
        if self.enabled and len(self.gelu_input_ranges) < 2000:
            self.gelu_input_ranges.append({
                'min':  x.min().item(),
                'max':  x.max().item(),
                'mean': x.mean().item(),
                'std':  x.std().item(),
            })

    def get_statistics(self):
        stats = {}

        for layer_name, scales in self.layer_input_scales.items():
            if scales:
                arr = np.array(scales)
                stats[layer_name] = {
                    'mean':    float(arr.mean()),
                    'median':  float(np.median(arr)),
                    'std':     float(arr.std()),
                    'min':     float(arr.min()),
                    'max':     float(arr.max()),
                    'p95':     float(np.percentile(arr, 95)),
                    'samples': len(arr),
                }

        if self.gelu_input_ranges:
            all_mins  = [r['min']  for r in self.gelu_input_ranges]
            all_maxs  = [r['max']  for r in self.gelu_input_ranges]
            all_means = [r['mean'] for r in self.gelu_input_ranges]
            all_stds  = [r['std']  for r in self.gelu_input_ranges]
            stats['gelu_input_range'] = {
                'overall_min':       float(np.min(all_mins)),
                'overall_max':       float(np.max(all_maxs)),
                'typical_min_p05':   float(np.percentile(all_mins, 5)),
                'typical_max_p95':   float(np.percentile(all_maxs, 95)),
                'mean_of_means':     float(np.mean(all_means)),
                'mean_of_stds':      float(np.mean(all_stds)),
                'median_min':        float(np.median(all_mins)),
                'median_max':        float(np.median(all_maxs)),
                'samples':           len(self.gelu_input_ranges),
            }

        return stats

    def enable(self):  self.enabled = True
    def disable(self): self.enabled = False


# Global instance
scale_tracker = ScaleTracker()


# ==============================================================================
# QUANTIZATION  — PER-TENSOR (hardware-friendly)
class Quantize(torch.autograd.Function):
    """
    Per-tensor symmetric INT8 fake-quantization with Straight-Through Estimator.
    Per-tensor means one global scale for the whole activation tensor.
    """
    @staticmethod
    def forward(ctx, x, num_bits=8):
        qmax = 2 ** (num_bits - 1) - 1          # 127 for INT8
        if x.numel() == 0:
            return x

        # ── Per-tensor scale (one value for the whole tensor) ─────────────────
        max_val = x.abs().max()
        scale   = (max_val / qmax).clamp(min=1e-8)   # clamp avoids div-by-zero

        q = torch.clamp((x / scale).round(), -qmax, qmax)
        return q * scale

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output, None   # STE: pass gradient through unchanged


# ==============================================================================
# QUANTIZED LINEAR LAYER

class QuantizedLinear(nn.Module):
    """
    Linear layer with per-tensor fake-quantization on both input and weight.
    Records input scale for hardware calibration during evaluation.
    """

    def __init__(self, in_features, out_features, bias=True, layer_name=''):
        super().__init__()
        self.in_features  = in_features
        self.out_features = out_features
        self.layer_name   = layer_name
        self.weight = nn.Parameter(torch.Tensor(out_features, in_features))
        if bias:
            self.bias = nn.Parameter(torch.Tensor(out_features))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in)
            nn.init.uniform_(self.bias, -bound, bound)

    def forward(self, input):
        # Record per-tensor input scale for hardware register calibration
        if scale_tracker.enabled:
            with torch.no_grad():
                abs_max     = input.abs().max().item()
                input_scale = abs_max / 127.0 if abs_max > 0 else 1.0
                scale_tracker.record_input_scale(self.layer_name, input_scale)

        q_input  = Quantize.apply(input)         # per-tensor fake-quant
        q_weight = Quantize.apply(self.weight)   # per-tensor fake-quant
        return F.linear(q_input, q_weight, self.bias)


# ==============================================================================
# FIXED-POINT ARITHMETIC

def to_fixed_point(x, bits, frac_bits):
    """Snap tensor to Qbits.frac_bits fixed-point grid."""
    scale   = 2.0 ** frac_bits
    min_val = -(2.0 ** (bits - 1))
    max_val =  (2.0 ** (bits - 1)) - 1
    fixed   = torch.clamp((x * scale).round(), min_val, max_val)
    return fixed / scale


# ==============================================================================
# PLA SOFTMAX  (piecewise-linear exp approximation)
# ==============================================================================

class PLASoftmax(nn.Module):
    def __init__(self, num_intervals=12, domain_min=-10.0, domain_max=0.0):
        super().__init__()
        self.domain_min = domain_min
        self.domain_max = domain_max
        coeffs_np = self._build_pla_coeffs(num_intervals, domain_min, domain_max)
        key_map = {'coeffs_m': 'm', 'coeffs_c': 'c', 'intervals': 'a'}
        for key in key_map:
            self.register_buffer(
                key,
                torch.tensor([c[key_map[key]] for c in coeffs_np], dtype=torch.float32)
            )

    @staticmethod
    def _build_pla_coeffs(num_intervals, domain_min, domain_max):
        xs        = np.linspace(domain_min, domain_max, 1001)
        ys        = np.exp(xs)
        intervals = np.linspace(domain_min, domain_max, num_intervals + 1)
        coeffs    = []
        for i in range(num_intervals):
            a, b = intervals[i], intervals[i + 1]
            mask = (xs >= a) & (xs <= b)
            m, c = np.polyfit(xs[mask], ys[mask], 1)
            coeffs.append({'a': a, 'm': m, 'c': c})
        return coeffs

    def pla_exp(self, x):
        x_clamped = torch.clamp(x, self.domain_min, self.domain_max)
        indices   = torch.clamp(
            torch.sum(x_clamped.unsqueeze(-1) >= self.intervals, dim=-1) - 1,
            0, len(self.intervals) - 2
        )
        return self.coeffs_m[indices] * x_clamped + self.coeffs_c[indices]

    def forward(self, scores):
        max_scores, _     = scores.max(dim=-1, keepdim=True)
        shifted           = scores - max_scores
        shifted_fx        = to_fixed_point(shifted, 32, 26)   # Q5.26 fixed-point
        exps              = self.pla_exp(shifted_fx)
        softmax_out       = exps / (exps.sum(dim=-1, keepdim=True) + 1e-9)
        return Quantize.apply(softmax_out)   # per-tensor quantize output


# ==============================================================================
# APPROXIMATE LAYER NORM

class ApproximateLayerNorm(nn.Module):
    def __init__(self, normalized_shape, eps=1e-5,
                 elementwise_affine=True, nr_iterations=8):
        super().__init__()
        self.normalized_shape   = normalized_shape
        self.eps                = eps
        self.elementwise_affine = elementwise_affine
        self.nr_iterations      = nr_iterations
        if self.elementwise_affine:
            self.weight = nn.Parameter(torch.ones(normalized_shape))
            self.bias   = nn.Parameter(torch.zeros(normalized_shape))

    def _sqrt_newton_raphson(self, S):
        x = torch.where(S > 1.0, S * 0.5, torch.ones_like(S))
        for _ in range(self.nr_iterations):
            x = 0.5 * (x + S / (x + 1e-9))
        return x

    def forward(self, x):
        mean   = x.mean(dim=-1, keepdim=True)
        var    = x.var(dim=-1,  keepdim=True, unbiased=False)
        var_fx = to_fixed_point(var, 32, 26)       # Q5.26 for sqrt unit
        std    = self._sqrt_newton_raphson(var_fx)
        x      = (x - mean) / (std + self.eps)
        if self.elementwise_affine:
            x = self.weight * x + self.bias
        return x


# ==============================================================================
# GCU — GELU COMPUTE UNIT

class ExponentialUnit(nn.Module):
    def __init__(self, num_segments=8):
        super().__init__()
        self.num_segments = num_segments
        coeffs = []
        for i in range(num_segments):
            x_s, x_e = i / num_segments, (i + 1) / num_segments
            y_s, y_e = 2 ** x_s, 2 ** x_e
            K = (y_e - y_s) / (x_e - x_s)
            B = y_s - K * x_s
            coeffs.append([K, B])
        self.register_buffer('coefficients',
                             torch.tensor(coeffs, dtype=torch.float32))

    def forward(self, x, use_log2e_scaling=True):
        if use_log2e_scaling:
            x = x * (1.0 + 0.5 - 0.0625)    # ≈ log2(e)
        x_int   = torch.floor(x).long()
        x_frac  = torch.clamp(x - x_int.float(), 0, 0.999999)
        seg_idx = torch.clamp(
            torch.floor(x_frac * self.num_segments).long(),
            0, self.num_segments - 1
        )
        K = self.coefficients[seg_idx, 0]
        B = self.coefficients[seg_idx, 1]
        return (K * x_frac + B) * (2.0 ** torch.clamp(x_int, -15, 15).float())


class DivisionUnit(nn.Module):
    def __init__(self):
        super().__init__()
        self.eu = ExponentialUnit()

    def leading_one_detector(self, x):
        abs_x  = torch.clamp(x.abs(), min=1e-8)
        log2_x = torch.log2(abs_x)
        w      = torch.floor(log2_x).long()
        m      = (log2_x - w.float()) + 1.0
        return w, m

    def forward(self, numerator, denominator, add_one_to_denominator=False):
        if add_one_to_denominator:
            denominator = 1.0 + denominator
        w1, m1   = self.leading_one_detector(numerator)
        w2, m2   = self.leading_one_detector(denominator)
        exponent = (m1 + w1.float()) - (m2 + w2.float())
        result   = self.eu(exponent, use_log2e_scaling=False)
        return result * (torch.sign(numerator) * torch.sign(denominator))


class PolynomialUnit(nn.Module):
    def __init__(self):
        super().__init__()
        self.sqrt_2_over_pi = 0.8
        self.cubic_coeff    = 0.03125 + 0.03125
        self.s_x_coeff      = -10.0 - 0.25 - 0.0625

    def forward(self, x):
        h_x = self.sqrt_2_over_pi * x + self.cubic_coeff * (x ** 3)
        return self.s_x_coeff * h_x


class GCU(nn.Module):
    """GELU Compute Unit — hardware polynomial approximation."""
    def __init__(self):
        super().__init__()
        self.polynomial_unit = PolynomialUnit()
        self.eu              = ExponentialUnit()
        self.du              = DivisionUnit()

    def forward(self, x):
        if scale_tracker.enabled:
            with torch.no_grad():
                scale_tracker.record_gelu_input(x)
        s_x      = self.polynomial_unit(x)
        exp_term = self.eu(-s_x, use_log2e_scaling=False)
        return self.du(x, exp_term, add_one_to_denominator=True)


# ==============================================================================
# MODEL OUTPUT DATACLASS

@dataclass
class SequenceClassifierOutput:
    loss:   Optional[torch.Tensor] = None
    logits: Optional[torch.Tensor] = None


# ==============================================================================
# BERT EMBEDDINGS
class BertEmbeddings(nn.Module):
    def __init__(self, vocab_size, hidden_size=768,
                 max_position_embeddings=512, type_vocab_size=2, dropout=0.1):
        super().__init__()
        self.word_embeddings       = nn.Embedding(vocab_size, hidden_size)
        self.position_embeddings   = nn.Embedding(max_position_embeddings, hidden_size)
        self.token_type_embeddings = nn.Embedding(type_vocab_size, hidden_size)
        self.LayerNorm             = ApproximateLayerNorm(hidden_size, eps=1e-12)
        self.dropout               = nn.Dropout(dropout)

    def forward(self, input_ids, token_type_ids=None):
        seq_len = input_ids.size(1)
        if token_type_ids is None:
            token_type_ids = torch.zeros_like(input_ids)
        pos_ids = (torch.arange(seq_len, dtype=torch.long, device=input_ids.device)
                       .unsqueeze(0).expand_as(input_ids))
        embeddings = (self.word_embeddings(input_ids)
                      + self.position_embeddings(pos_ids)
                      + self.token_type_embeddings(token_type_ids))
        return self.dropout(self.LayerNorm(embeddings))


# ==============================================================================
# MULTI-HEAD SELF-ATTENTION
# ==============================================================================

class MultiHeadSelfAttention(nn.Module):
    def __init__(self, hidden_size=768, num_attention_heads=12,
                 dropout=0.1, layer_id=0):
        super().__init__()
        self.num_heads = num_attention_heads
        self.head_dim  = hidden_size // num_attention_heads
        self.scale     = self.head_dim ** -0.5

        base = f'bert.encoder.layers.{layer_id}.attention'
        self.q       = QuantizedLinear(hidden_size, hidden_size,
                                       layer_name=f'{base}.q')
        self.k       = QuantizedLinear(hidden_size, hidden_size,
                                       layer_name=f'{base}.k')
        self.v       = QuantizedLinear(hidden_size, hidden_size,
                                       layer_name=f'{base}.v')
        self.out     = QuantizedLinear(hidden_size, hidden_size,
                                       layer_name=f'{base}.out')
        self.softmax       = PLASoftmax()
        self.attn_dropout  = nn.Dropout(dropout)
        self.proj_dropout  = nn.Dropout(dropout)

    def transpose_for_scores(self, x):
        return (x.view(x.size()[:-1] + (self.num_heads, self.head_dim))
                  .permute(0, 2, 1, 3))

    def forward(self, hidden_states, attention_mask=None):
        TILE_SIZE = 32

        # ── Q / K / V GeMM (INT8×INT8 → INT32) → Quantize → INT8 buffer
        # One per-tensor Quantize.apply after the full projection (not per tile).
        # This is the INT8 output buffer the hardware writes after the GeMM+bias.
        # All tiles downstream read from this same buffer with the same scale.
        q_full = Quantize.apply(self.transpose_for_scores(self.q(hidden_states)))
        k_full = Quantize.apply(self.transpose_for_scores(self.k(hidden_states)))
        v_full = Quantize.apply(self.transpose_for_scores(self.v(hidden_states)))

        B, H, Sq, D = q_full.shape
        Sk           = k_full.size(2)

        # ── QKᵀ GeMM (INT8×INT8 → INT32) → Dequant→Q5.26 → Softmax → INT8
        # Tiles read from the already-INT8 Q/K buffers — no re-quantization.
        # to_fixed_point(Q5.26) inside PLASoftmax simulates the dequant→Q5.26 step.
        # PLASoftmax.forward() ends with Quantize.apply → INT8 (Q1.15→INT8).
        attn_scores = torch.zeros(B, H, Sq, Sk, device=hidden_states.device)
        for i in range(0, Sq, TILE_SIZE):
            for j in range(0, Sk, TILE_SIZE):
                attn_scores[:, :, i:i+TILE_SIZE, j:j+TILE_SIZE] = (
                    torch.matmul(
                        q_full[:, :, i:i+TILE_SIZE, :],
                        k_full[:, :, j:j+TILE_SIZE, :].transpose(-1, -2)
                    ) * self.scale
                )

        if attention_mask is not None:
            attn_scores = attn_scores + attention_mask

        # PLASoftmax: INT32 scores → Q5.26 → PLA-exp → softmax → Quantize → INT8
        attn_probs = self.attn_dropout(self.softmax(attn_scores))

        # ── ·V GeMM (INT8×INT8 → INT32) → Quantize → INT8 
        # V is already INT8 from the buffer above.
        # After the weighted-sum GeMM the hardware quantizes the INT32 accumulator
        # back to INT8 before feeding into the Wo GeMM.
        context = torch.zeros_like(v_full)
        for i in range(0, Sq, TILE_SIZE):
            context[:, :, i:i+TILE_SIZE, :] = torch.matmul(
                attn_probs[:, :, i:i+TILE_SIZE, :], v_full
            )

        # Quantize ·V output → INT8  (PDF: Quantize 32→INT8 after ·V GeMM)
        context = Quantize.apply(
            context.permute(0, 2, 1, 3).contiguous().view(hidden_states.size())
        )

        # ── Wo GeMM (INT8×INT8 → INT32) → Bias(INT32) → Quantize → INT8
        return self.proj_dropout(Quantize.apply(self.out(context)))


# ==============================================================================
# FEED-FORWARD BLOCK
class FeedForward(nn.Module):
    def __init__(self, hidden_size=768, intermediate_size=3072,
                 dropout=0.1, layer_id=0):
        super().__init__()
        base = f'bert.encoder.layers.{layer_id}.ffn'
        self.dense_1 = QuantizedLinear(hidden_size, intermediate_size,
                                       layer_name=f'{base}.dense_1')
        self.dense_2 = QuantizedLinear(intermediate_size, hidden_size,
                                       layer_name=f'{base}.dense_2')
        self.dropout = nn.Dropout(dropout)
        self.gcu     = GCU()

    def forward(self, x):
        TILE_SIZE = 32
        B, S, _ = x.shape

        # ── ffn_1 GeMM (INT8×INT8 → INT32) → Dequant→Q10.22 → GELU → Q48.12→INT8
        # to_fixed_point inside GCU handles the Q10.22 dequant step.
        # Quantize.apply after GCU simulates Q48.12→INT8 at the GCU output register.
        out1 = torch.zeros(B, S, self.dense_1.out_features, device=x.device)
        for i in range(0, S, TILE_SIZE):
            out1[:, i:i+TILE_SIZE, :] = self.dense_1(x[:, i:i+TILE_SIZE, :])

        x = Quantize.apply(self.gcu(out1))   # Q48.12 → INT8

        # ── ffn_2 GeMM (INT8×INT8 → INT32) → Bias(INT32) → Quantize → INT8
        out2 = torch.zeros(B, S, self.dense_2.out_features, device=x.device)
        for i in range(0, S, TILE_SIZE):
            out2[:, i:i+TILE_SIZE, :] = self.dense_2(x[:, i:i+TILE_SIZE, :])

        # Quantize ffn_2 output → INT8
        return self.dropout(Quantize.apply(out2))


# ==============================================================================
# TRANSFORMER ENCODER LAYER
class TransformerEncoderLayer(nn.Module):
    def __init__(self, hidden_size=768, num_attention_heads=12,
                 intermediate_size=3072, dropout=0.1, layer_id=0):
        super().__init__()
        self.attention      = MultiHeadSelfAttention(hidden_size,
                                                     num_attention_heads,
                                                     dropout, layer_id=layer_id)
        self.attn_layer_norm = ApproximateLayerNorm(hidden_size, eps=1e-12)
        self.ffn             = FeedForward(hidden_size, intermediate_size,
                                           dropout, layer_id=layer_id)
        self.ffn_layer_norm  = ApproximateLayerNorm(hidden_size, eps=1e-12)

    def forward(self, x, attention_mask=None):
        attn_output  = self.attention(x, attention_mask=attention_mask)

        # ── Residual Add → Dequant→Q5.26 → LayerNorm → INT8
        #      → LayerNorm → Quantize→INT8
        # Both x and attn_output are already INT8 from their respective
        # Quantize.apply outputs (Wo output and encoder input).
        # to_fixed_point(Q5.26) inside ApproximateLayerNorm handles the dequant step.
        # Quantize.apply after LayerNorm = LayerNorm output → INT8 register.
        norm_input_1 = Quantize.apply(x) + Quantize.apply(attn_output)
        x            = Quantize.apply(self.attn_layer_norm(norm_input_1))  # LN1 → INT8

        ffn_output   = self.ffn(x)

        # ── Residual Add → Dequant→Q5.26 → LayerNorm → INT8
        #      → LayerNorm → Quantize→INT8
        #
        # x is INT8 from LN1 above. ffn_output is INT8 from ffn_2 Quantize.apply.
        norm_input_2 = Quantize.apply(x) + Quantize.apply(ffn_output)
        x            = Quantize.apply(self.ffn_layer_norm(norm_input_2))   # LN2 → INT8
        return x


# ==============================================================================
# TRANSFORMER ENCODER

class TransformerEncoder(nn.Module):
    def __init__(self, num_hidden_layers=12, **layer_kwargs):
        super().__init__()
        self.layers = nn.ModuleList([
            TransformerEncoderLayer(**layer_kwargs, layer_id=i)
            for i in range(num_hidden_layers)
        ])

    def forward(self, x, attention_mask=None):
        for layer in self.layers:
            x = layer(x, attention_mask=attention_mask)
        return x


# ==============================================================================
# BERT MODEL

class BertModel(nn.Module):
    def __init__(self, vocab_size, hidden_size=768, num_hidden_layers=12,
                 num_attention_heads=12, intermediate_size=3072,
                 max_position_embeddings=512, type_vocab_size=2, dropout=0.1):
        super().__init__()
        self.embeddings = BertEmbeddings(vocab_size, hidden_size,
                                         max_position_embeddings,
                                         type_vocab_size, dropout)
        self.encoder    = TransformerEncoder(
            num_hidden_layers    = num_hidden_layers,
            hidden_size          = hidden_size,
            num_attention_heads  = num_attention_heads,
            intermediate_size    = intermediate_size,
            dropout              = dropout,
        )
        # Pooler is standard float linear (not quantized — matches original)
        self.pooler            = nn.Linear(hidden_size, hidden_size)
        self.pooler_activation = nn.Tanh()

    def forward(self, input_ids, token_type_ids=None, attention_mask=None):
        embeddings    = self.embeddings(input_ids, token_type_ids=token_type_ids)
        extended_mask = (
            (1.0 - attention_mask.unsqueeze(1).unsqueeze(2)) * -10000.0
            if attention_mask is not None else None
        )
        encoder_output = self.encoder(embeddings, attention_mask=extended_mask)
        pooled         = self.pooler_activation(
            self.pooler(encoder_output[:, 0])
        )
        return encoder_output, pooled


# ==============================================================================
# BERT FOR SEQUENCE CLASSIFICATION

class BertForSequenceClassification(nn.Module):
    def __init__(self, bert: BertModel, num_labels=2, dropout=0.1):
        super().__init__()
        self.bert       = bert
        self.dropout    = nn.Dropout(dropout)
        self.classifier = QuantizedLinear(bert.pooler.in_features, num_labels,
                                          layer_name='classifier')

    def forward(self, input_ids, token_type_ids=None,
                attention_mask=None, labels=None):
        _, pooled_output = self.bert(input_ids,
                                     token_type_ids=token_type_ids,
                                     attention_mask=attention_mask)
        logits = self.classifier(self.dropout(pooled_output))
        loss   = (F.cross_entropy(logits.view(-1, self.classifier.out_features),
                                  labels.view(-1))
                  if labels is not None else None)
        return SequenceClassifierOutput(loss=loss, logits=logits)


# ==============================================================================
# WEIGHT EXTRACTION
def extract_and_save_quantized_weights(model: nn.Module, file_path: str):
    """
    Weights:  INT8  + per-tensor weight_scale
    Biases:   INT32 + bias_scale  (= input_scale × weight_scale)
    Scales:   collected by ScaleTracker during evaluation (per-tensor max)
    """
    import json

    torch.cuda.empty_cache()
    quantized_state_dict = {}

    print("\n" + "=" * 70)
    print("EXTRACTING WEIGHTS AND SCALES FOR HARDWARE")
    print("=" * 70)

    scale_stats = scale_tracker.get_statistics()

    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue

        param_data = param.detach().cpu().numpy()

        # ── Weights (2D matrices → INT8) ──────────────────────────────────────
        if 'weight' in name and param_data.ndim == 2:
            abs_max      = np.abs(param_data).max()
            weight_scale = abs_max / 127.0 if abs_max > 0 else 1.0

            q_weight = np.round(param_data / weight_scale).clip(-128, 127).astype(np.int8)
            quantized_state_dict[name]               = q_weight
            quantized_state_dict[f'{name}.scale']    = np.float32(weight_scale)

            # Per-tensor input scale from ScaleTracker
            layer_key = name.replace('.weight', '')
            if layer_key in scale_stats and 'median' in scale_stats[layer_key]:
                input_scale = scale_stats[layer_key]['median']
                quantized_state_dict[f'{name}.input_scale'] = np.float32(input_scale)
                print(f"  {name}: weight_scale={weight_scale:.6e}, "
                      f"input_scale={input_scale:.6e}")
            else:
                quantized_state_dict[f'{name}.input_scale'] = np.float32(1.0)
                print(f"  {name}: weight_scale={weight_scale:.6e}, "
                      f"input_scale=1.0 (default)")

        # ── Biases (1D → INT32, scale = input_scale × weight_scale) ──────────
        elif 'bias' in name and param_data.ndim == 1:
            weight_name     = name.replace('.bias', '.weight')
            weight_scale_key = f'{weight_name}.scale'
            input_scale_key  = f'{weight_name}.input_scale'

            if (weight_scale_key in quantized_state_dict and
                    input_scale_key in quantized_state_dict):
                weight_scale = float(quantized_state_dict[weight_scale_key])
                input_scale  = float(quantized_state_dict[input_scale_key])
                bias_scale   = input_scale * weight_scale

                bias_int32   = np.round(param_data / bias_scale).clip(
                    -2147483648, 2147483647).astype(np.int32)

                quantized_state_dict[name]             = bias_int32
                quantized_state_dict[f'{name}.scale']  = np.float32(bias_scale)
                print(f"  {name}: INT32, bias_scale={bias_scale:.6e}")
            else:
                # Fallback (e.g. pooler bias — pooler is float, not QuantizedLinear)
                quantized_state_dict[name] = param_data.astype(np.float32)
                print(f"  {name}: FP32 (fallback)")

        # ── Other parameters (embeddings 2D, LayerNorm, etc.) ─────────────────
        else:
            if param_data.ndim >= 2:
                abs_max = np.abs(param_data).max()
                scale   = abs_max / 127.0 if abs_max > 0 else 1.0
                q_param = np.round(param_data / scale).clip(-128, 127).astype(np.int8)
                quantized_state_dict[name]            = q_param
                quantized_state_dict[f'{name}.scale'] = np.float32(scale)
            else:
                quantized_state_dict[name] = param_data.astype(np.float32)

    # Save .npz
    np.savez_compressed(file_path, **quantized_state_dict)
    print(f"\n  Saved quantized weights to: {file_path}")

    # Save scale metadata JSON
    metadata_file = file_path.replace('.npz', '_scale_metadata.json')
    with open(metadata_file, 'w') as f:
        json.dump(scale_stats, f, indent=2)
    print(f"  Saved scale metadata to:    {metadata_file}")

    # GELU input range analysis (hardware design guidance)
    if 'gelu_input_range' in scale_stats:
        g = scale_stats['gelu_input_range']
        print("\n" + "=" * 70)
        print("GELU INPUT RANGE ANALYSIS (CRITICAL FOR HARDWARE!)")
        print("=" * 70)
        print(f"\n  Measured from {g['samples']} batches:")
        print(f"  Overall range:       [{g['overall_min']:.4f}, {g['overall_max']:.4f}]")
        print(f"  Typical range (90%): [{g['typical_min_p05']:.4f}, {g['typical_max_p95']:.4f}]")
        print(f"  Median range:        [{g['median_min']:.4f}, {g['median_max']:.4f}]")

        max_abs   = max(abs(g['overall_min']), abs(g['overall_max']))
        int_bits  = int(np.ceil(np.log2(max_abs + 1))) + 1
        frac_bits = 16 - int_bits
        print(f"\n  Recommended fixed-point format: Q{int_bits}.{frac_bits}")
        print(f"  Range:     [{-2**(int_bits-1):.1f}, {2**(int_bits-1)-1:.1f}]")
        print(f"  Precision: {2**-frac_bits:.6f}")

        range_span = g['typical_max_p95'] - g['typical_min_p05']
        print(f"\n  PLA segments estimate:")
        for n in [8, 16, 32]:
            print(f"    {n:2d} segments: {range_span/n:.4f} per segment "
                  f"({n*4} bytes for slopes/intercepts)")
        print("=" * 70)

    # Extraction summary
    print("\n" + "=" * 70)
    print("EXTRACTION SUMMARY")
    print("=" * 70)
    total_mb = sum(v.nbytes for v in quantized_state_dict.values()) / (1024 ** 2)
    n_input_scales = sum(1 for k in quantized_state_dict if '.input_scale' in k)
    n_int32_bias   = sum(1 for k, v in quantized_state_dict.items()
                         if 'bias' in k and hasattr(v, 'dtype')
                         and v.dtype == np.int32)
    print(f"  Total size:                 {total_mb:.2f} MB")
    print(f"  Layers with input scales:   {n_input_scales}")
    print(f"  INT32 biases:               {n_int32_bias}")
    print("=" * 70)


# ==============================================================================
# MAIN TRAINING LOOP
# ==============================================================================

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    NUM_EPOCHS     = 3
    BATCH_SIZE     = 16
    LEARNING_RATE  = 5e-5
    MAX_SEQ_LENGTH = 128

    tokenizer    = BertTokenizer.from_pretrained('bert-base-uncased')
    raw_datasets = load_dataset("glue", "sst2")

    def tokenize_function(examples):
        return tokenizer(examples["sentence"], padding="max_length",
                         truncation=True, max_length=MAX_SEQ_LENGTH)

    tokenized_datasets = (
        raw_datasets
        .map(tokenize_function, batched=True)
        .remove_columns(["sentence", "idx"])
        .rename_column("label", "labels")
    )
    tokenized_datasets.set_format("torch")

    train_dataloader = DataLoader(tokenized_datasets["train"],
                                  shuffle=True, batch_size=BATCH_SIZE)
    eval_dataloader  = DataLoader(tokenized_datasets["validation"],
                                  batch_size=BATCH_SIZE)

    model = BertForSequenceClassification(
        bert=BertModel(
            vocab_size               = tokenizer.vocab_size,
            hidden_size              = 768,
            num_hidden_layers        = 12,
            num_attention_heads      = 12,
            intermediate_size        = 3072,
            max_position_embeddings  = 512,
        ),
        num_labels=2,
    ).to(device)

    print(f"\nBERT-base Hardware-Aware Model  "
          f"({sum(p.numel() for p in model.parameters()):,} parameters)")
    print("Quantization: PER-TENSOR (hardware-accurate)\n")

    optimizer            = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE)
    num_training_steps   = NUM_EPOCHS * len(train_dataloader)
    lr_scheduler         = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps  = int(0.1 * num_training_steps),
        num_training_steps = num_training_steps,
    )

    accuracy_metric = evaluate.load("accuracy")
    progress_bar    = tqdm(range(num_training_steps))

    for epoch in range(NUM_EPOCHS):

        # ── Training ──────────────────────────────────────────────────────────
        model.train()
        scale_tracker.disable()

        for batch in train_dataloader:
            batch = {k: v.to(device) for k, v in batch.items()}
            loss  = model(**batch).loss
            loss.backward()
            optimizer.step()
            lr_scheduler.step()
            optimizer.zero_grad()
            progress_bar.update(1)

        # ── Evaluation + scale collection ─────────────────────────────────────
        print(f"\n--- Epoch {epoch + 1}/{NUM_EPOCHS} ---")
        print("Evaluating and collecting activation scales...")
        model.eval()
        scale_tracker.enable()
        accuracy_metric = evaluate.load("accuracy")

        with torch.no_grad():
            for i, batch in enumerate(eval_dataloader):
                batch       = {k: v.to(device) for k, v in batch.items()}
                logits      = model(**batch).logits
                predictions = torch.argmax(logits, dim=-1)
                accuracy_metric.add_batch(predictions=predictions,
                                          references=batch["labels"])
                if i >= 100:   # collect from first 100 batches (~1600 samples)
                    break

        scale_tracker.disable()
        result = accuracy_metric.compute()
        print(f"Validation Accuracy: {result['accuracy']:.4f}")

    # ── Save weights + scales ─────────────────────────────────────────────────
    print("\nTraining complete. Extracting weights and scales...")
    output_weights_file = '/kaggle/working/weights_with_scales.npz'   # ← change path
    extract_and_save_quantized_weights(model, output_weights_file)


if __name__ == "__main__":
    main()