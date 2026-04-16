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
    Tracks per-tensor activation scales at EVERY Quantize.apply site.

    Two collections:
      layer_input_scales  — input scales recorded inside QuantizedLinear
                            (one entry per forward call, keyed by layer name)
      activation_scales   — output scales at every other quantization point:
                            Q/K/V outputs, softmax, ·V context, Wo, GCU,
                            ffn2, residual operands, LN1, LN2 per layer.

    Both are saved to the .npz so the hardware has static scales for every
    register that needs programming before inference.
    """
    def __init__(self):
        self.layer_input_scales = defaultdict(list)
        self.activation_scales  = defaultdict(list)   # keyed by full node name
        self.gelu_input_ranges  = []
        self.enabled = False

    def record_input_scale(self, layer_name, scale):
        if self.enabled and len(self.layer_input_scales[layer_name]) < 500:
            self.layer_input_scales[layer_name].append(scale)

    def record_activation_scale(self, name: str, x: torch.Tensor):
        """Record max(|x|)/127 for a tensor at a Quantize.apply site."""
        if self.enabled and len(self.activation_scales[name]) < 500:
            with torch.no_grad():
                abs_max = x.detach().abs().max().item()
                self.activation_scales[name].append(
                    abs_max / 127.0 if abs_max > 1e-8 else 1e-8
                )

    def record_gelu_input(self, x):
        if self.enabled and len(self.gelu_input_ranges) < 2000:
            self.gelu_input_ranges.append({
                'min':  x.min().item(),
                'max':  x.max().item(),
                'mean': x.mean().item(),
                'std':  x.std().item(),
            })

    def _summarise(self, scales):
        arr = np.array(scales)
        return {
            'mean':    float(arr.mean()),
            'median':  float(np.median(arr)),
            'std':     float(arr.std()),
            'min':     float(arr.min()),
            'max':     float(arr.max()),
            'p95':     float(np.percentile(arr, 95)),
            'samples': len(arr),
        }

    def get_statistics(self):
        stats = {}

        for layer_name, scales in self.layer_input_scales.items():
            if scales:
                stats[layer_name] = self._summarise(scales)

        for name, scales in self.activation_scales.items():
            if scales:
                stats[name] = self._summarise(scales)

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
    Bias is quantized to INT8 (bias_scale = input_scale × weight_scale,
    clamped to the INT8 range [-128, 127]).
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

        q_input  = Quantize.apply(input)         # per-tensor fake-quant → INT8
        q_weight = Quantize.apply(self.weight)   # per-tensor fake-quant → INT8

        # ── Bias fake-quantization to INT8 ────────────────────────────────────
        # bias_scale = input_scale × weight_scale (mirrors hardware accumulation)
        if self.bias is not None:
            with torch.no_grad():
                w_abs_max    = self.weight.abs().max().item()
                w_scale      = w_abs_max / 127.0 if w_abs_max > 0 else 1.0
                in_abs_max   = input.abs().max().item()
                in_scale     = in_abs_max / 127.0 if in_abs_max > 0 else 1.0
                bias_scale   = in_scale * w_scale
            # Snap bias to INT8 grid (fake-quant, keeps float dtype for autograd)
            q_bias = Quantize.apply(self.bias / bias_scale) * bias_scale  \
                     if bias_scale > 1e-12 else self.bias
        else:
            q_bias = None

        return F.linear(q_input, q_weight, q_bias)


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
        max_scores, _ = scores.max(dim=-1, keepdim=True)
        shifted       = scores - max_scores
        shifted_fx    = to_fixed_point(shifted, 32, 26)       # QKt INT32 → Q5.26
        exps          = self.pla_exp(shifted_fx)
        softmax_out   = exps / (exps.sum(dim=-1, keepdim=True) + 1e-9)
        # PDF: "Softmax Quantization Q1.15 to 8 bit"
        # Snap to Q1.15 grid before requantizing to INT8.
        softmax_q1_15 = to_fixed_point(softmax_out, 16, 15)   # Q1.15 intermediate
        return Quantize.apply(softmax_q1_15)                   # Q1.15 → INT8


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
# Replaced with RTL-accurate pipeline (EU1 → DU → EU2) from PLA_EU / GELU.sv.

class ExponentialUnit(nn.Module):
    """
    Exponential Unit matching RTL EU.sv pipeline and PLA_EU.py LUT.
    Stages: Extract int/frac → LUT lookup → Mantissa (K·x_frac + B) →
    Q10.22→Q48.16 conversion → Asymmetric barrel shifter
    """
    def __init__(self, num_segments=8):
        super().__init__()
        self.num_segments = num_segments
        self.Q_IN  = 22          # Input fractional bits  (Q10.22)
        self.Q_OUT = 16          # Output fractional bits (Q48.16 in 64-bit)
        self.Q_SHIFT = self.Q_IN - self.Q_OUT  # = 6 bits for Q22→Q16 conversion

        # ── Generate K and B coefficients using PLA_EU.py algorithm ──────────
        k_coeffs = []
        b_coeffs = []
        segment_size = 1.0 / num_segments
        for i in range(num_segments):
            x_start = i * segment_size
            x_end   = (i + 1) * segment_size
            y_start = 2 ** x_start
            y_end   = 2 ** x_end
            K = (y_end - y_start) / segment_size   # slope
            B = y_start - K * x_start               # intercept
            k_coeffs.append(K)
            b_coeffs.append(B)

        self.register_buffer('k_lut', torch.tensor(k_coeffs, dtype=torch.float32))
        self.register_buffer('b_lut', torch.tensor(b_coeffs, dtype=torch.float32))

    def forward(self, x, use_log2e_scaling=True):
        """
        RTL Pipeline matching EU.sv:
        1. Apply log2(e) scaling if needed
        2. Extract integer part
        3. Extract fractional part
        4. Get segment index from top 3 bits of fractional part
        5. LUT lookup: K, B = LUT[segment_index]
        6. Compute mantissa: K * x_frac + B  (Q10.22 equivalent)
        7. Asymmetric barrel shifter: left-shift if s_int >= 0, else right-shift
        Output is in Q48.16 format.
        """
        # Stage 1: Apply log2(e) scaling
        if use_log2e_scaling:
            x = x * math.log2(math.e)  # precise log2(e) ≈ 1.4427

        # Stage 2–3: Extract integer and fractional parts
        x_int  = torch.floor(x).long()
        x_frac = torch.clamp(x - x_int.float(), 0, 0.999999)

        # Stage 4: Segment index from top bits of fractional part → [0, num_segments)
        seg_idx = torch.clamp(
            torch.floor(x_frac * self.num_segments).long(),
            0, self.num_segments - 1
        )

        # Stage 5: LUT lookup
        K = self.k_lut[seg_idx]  # slope
        B = self.b_lut[seg_idx]  # intercept

        # Stage 6: Mantissa = K * x_frac + B  (Q10.22 equivalent in float)
        mantissa = K * x_frac + B

        # Stage 7 (implicit): Q10.22 → Q48.16 right-shift by Q_SHIFT=6 is
        # represented naturally in floating-point; the barrel shifter below
        # applies the integer-part scaling.

        # Stage 8: Asymmetric barrel shifter
        #   s_int >= 0  →  mantissa << s_int  (left shift  = multiply by 2^s_int)
        #   s_int <  0  →  mantissa >> -s_int (right shift = divide  by 2^-s_int)
        x_int_clamped = torch.clamp(x_int, -15, 15)
        result = torch.where(
            x_int >= 0,
            mantissa * (2.0 ** x_int_clamped.float()),
            mantissa / (2.0 ** torch.clamp(-x_int_clamped, 0, 15).float())
        )
        return result   # Q48.16 format


class DivisionUnit(nn.Module):
    """
    Division Unit matching RTL DU.sv.
    Computes exponent = (m1 + s1) - (m2 + s2) for EU2 to evaluate.
    """
    def __init__(self):
        super().__init__()
        self.Q = 16   # fractional bits for Q48.16

    def leading_one_detector(self, x):
        """
        Leading One Detector: finds MSB position (w) and normalised mantissa (m).
        Matches RTL DU.sv calculation.
        Returns:
            w: MSB bit position (integer)
            m: Normalised mantissa in [1.0, 2.0)
        """
        abs_x  = torch.clamp(x.abs(), min=1e-8)
        log2_x = torch.log2(abs_x)
        w      = torch.floor(log2_x).long()
        m      = (log2_x - w.float()) + 1.0   # normalised mantissa
        return w, m

    def forward(self, numerator, denominator, add_one_to_denominator=False):
        """
        RTL: exponent = (m1 + s1) - (m2 + s2)
             s_i = (w_i - Q)
        Returns exponent (for EU2) and sign.
        """
        if add_one_to_denominator:
            denominator = 1.0 + denominator

        w1, m1 = self.leading_one_detector(numerator)
        w2, m2 = self.leading_one_detector(denominator)

        s1 = (w1.float() - self.Q)
        s2 = (w2.float() - self.Q)

        exponent = (m1 + s1) - (m2 + s2)
        sign     = torch.sign(numerator) * torch.sign(denominator)
        return exponent, sign


class PolynomialUnit(nn.Module):
    """
    Polynomial Unit: s(x) = -2·log2(e)·√(2/π)·(x + 0.044715·x³)
    Matches RTL PU.sv / GELU reference.
    """
    def __init__(self):
        super().__init__()
        self.log2_e         = math.log2(math.e)            # ≈ 1.4427
        self.sqrt_2_over_pi = math.sqrt(2.0 / math.pi)    # ≈ 0.7979
        self.coefficient    = -2.0 * self.log2_e * self.sqrt_2_over_pi  # ≈ -2.3046
        self.cubic_coeff    = 0.044715

    def forward(self, x):
        inner = x + self.cubic_coeff * (x ** 3)
        return self.coefficient * inner


class GCU(nn.Module):
    """
    GELU Compute Unit — RTL-accurate hardware pipeline (EU1 → DU → EU2).

    Input format:  Q10.22  (snapped by FeedForward before calling GCU)
    Output format: Q48.16  (EU2 output; snapped to INT8 by FeedForward after)

    Pipeline matching GELU.sv:
      Stage 1: PolynomialUnit  → s(x)
      Stage 2: EU1             → exp(s(x))          [Q48.16]
      Stage 3: DU              → exponent & sign     (x / (1 + exp(s(x))))
      Stage 4: EU2             → final result        [Q48.16]
    """
    def __init__(self):
        super().__init__()
        self.polynomial_unit = PolynomialUnit()
        self.eu1             = ExponentialUnit()   # Stage 2
        self.du              = DivisionUnit()      # Stage 3
        self.eu2             = ExponentialUnit()   # Stage 4

    def forward(self, x):
        # x is expected in Q10.22 format (enforced by FeedForward)
        if scale_tracker.enabled:
            with torch.no_grad():
                scale_tracker.record_gelu_input(x)

        # Stage 1: s(x) = -2·log2(e)·√(2/π)·(x + 0.044715·x³)
        s_x = self.polynomial_unit(x)

        # Stage 2: EU1 — compute exp(s(x))  (no additional log2(e) scaling)
        exp_term = self.eu1(s_x, use_log2e_scaling=False)

        # Stage 3: DU — exponent = (m1+s1) - (m2+s2), plus sign
        exponent, sign = self.du(x, exp_term, add_one_to_denominator=True)

        # Stage 4: EU2 — evaluate 2^exponent  (result in Q48.16)
        result = self.eu2(exponent, use_log2e_scaling=False)
        return result * sign   # Q48.16 output


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
class MultiHeadSelfAttention(nn.Module):
    def __init__(self, hidden_size=768, num_attention_heads=12,
                 dropout=0.1, layer_id=0):
        super().__init__()
        self.num_heads = num_attention_heads
        self.head_dim  = hidden_size // num_attention_heads
        self.scale     = self.head_dim ** -0.5

        base = f'bert.encoder.layers.{layer_id}.attention'
        self.base      = base
        self.layer_id  = layer_id
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
        b = self.base   # e.g. 'bert.encoder.layers.0.attention'

        # ── Q / K / V GeMM (INT8×INT8 → INT32) → Quantize → INT8 buffer
        q_raw  = self.transpose_for_scores(self.q(hidden_states))
        k_raw  = self.transpose_for_scores(self.k(hidden_states))
        v_raw  = self.transpose_for_scores(self.v(hidden_states))
        if scale_tracker.enabled:
            scale_tracker.record_activation_scale(f'{b}.q.output_scale',   q_raw)
            scale_tracker.record_activation_scale(f'{b}.k.output_scale',   k_raw)
            scale_tracker.record_activation_scale(f'{b}.v.output_scale',   v_raw)
        q_full = Quantize.apply(q_raw)
        k_full = Quantize.apply(k_raw)
        v_full = Quantize.apply(v_raw)

        B, H, Sq, D = q_full.shape
        Sk           = k_full.size(2)

        # ── QKᵀ GeMM (INT8×INT8 → INT32)
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

        # ── Quantize QxKᵀ INT32 → INT8
        if scale_tracker.enabled:
            scale_tracker.record_activation_scale(f'{b}.qkt_buffer.output_scale', attn_scores)
        attn_scores_int8 = Quantize.apply(attn_scores)

        # PLASoftmax: INT8 → de-quantise to Q5.26 → PLA-exp → softmax → INT8
        attn_probs_raw = self.softmax(attn_scores_int8)

        if scale_tracker.enabled:
            scale_tracker.record_activation_scale(f'{b}.softmax.output_scale', attn_probs_raw)
        attn_probs = self.attn_dropout(attn_probs_raw)

        # ── ·V GeMM (INT8×INT8 → INT32) → Quantize → INT8
        context = torch.zeros_like(v_full)
        for i in range(0, Sq, TILE_SIZE):
            context[:, :, i:i+TILE_SIZE, :] = torch.matmul(
                attn_probs[:, :, i:i+TILE_SIZE, :], v_full
            )

        context_merged = context.permute(0, 2, 1, 3).contiguous().view(hidden_states.size())
        if scale_tracker.enabled:
            scale_tracker.record_activation_scale(f'{b}.ctx.output_scale', context_merged)
        context = Quantize.apply(context_merged)

        # ── Wo GeMM (INT8×INT8 → INT32) → Bias(INT8) → Quantize → INT8
        wo_out = self.out(context)
        if scale_tracker.enabled:
            scale_tracker.record_activation_scale(f'{b}.out.output_scale', wo_out)
        return self.proj_dropout(Quantize.apply(wo_out))


# ==============================================================================
# FEED-FORWARD BLOCK
class FeedForward(nn.Module):
    def __init__(self, hidden_size=768, intermediate_size=3072,
                 dropout=0.1, layer_id=0):
        super().__init__()
        base = f'bert.encoder.layers.{layer_id}.ffn'
        self.base     = base
        self.layer_id = layer_id
        self.dense_1 = QuantizedLinear(hidden_size, intermediate_size,
                                       layer_name=f'{base}.dense_1')
        self.dense_2 = QuantizedLinear(intermediate_size, hidden_size,
                                       layer_name=f'{base}.dense_2')
        self.dropout = nn.Dropout(dropout)
        self.gcu     = GCU()

    def forward(self, x):
        TILE_SIZE = 32
        B, S, _ = x.shape

        # ── ffn_1 GeMM (INT8×INT8 → INT32, bias INT8)
        out1 = torch.zeros(B, S, self.dense_1.out_features, device=x.device)
        for i in range(0, S, TILE_SIZE):
            out1[:, i:i+TILE_SIZE, :] = self.dense_1(x[:, i:i+TILE_SIZE, :])

        # ── Quantize ffn_1 INT32 → INT8 intermediate buffer
        if scale_tracker.enabled:
            scale_tracker.record_activation_scale(
                f'{self.base}.intermediate_buffer.output_scale', out1)
        out1_int8 = Quantize.apply(out1)

        # ── De-quantise INT8 → Q10.22  (GCU input format)
        out1_q10_22 = to_fixed_point(out1_int8, 32, 22)    # Q10.22

        # ── GCU: Q10.22 input → Q48.16 output  (EU1 → DU → EU2 pipeline)
        gcu_out    = self.gcu(out1_q10_22)                  # Q48.16

        # ── Snap GCU output to Q48.16 grid (64-bit container, 16 frac bits)
        gcu_q48_16 = to_fixed_point(gcu_out, 64, 16)        # Q48.16
        if scale_tracker.enabled:
            scale_tracker.record_activation_scale(
                f'{self.base}.gcu.output_scale', gcu_q48_16)

        # ── Q48.16 → INT8  (requantize before ffn_2)
        x = Quantize.apply(gcu_q48_16)

        # ── ffn_2 GeMM (INT8×INT8 → INT32, bias INT8) → Quantize → INT8
        out2 = torch.zeros(B, S, self.dense_2.out_features, device=x.device)
        for i in range(0, S, TILE_SIZE):
            out2[:, i:i+TILE_SIZE, :] = self.dense_2(x[:, i:i+TILE_SIZE, :])

        if scale_tracker.enabled:
            scale_tracker.record_activation_scale(
                f'{self.base}.dense_2.output_scale', out2)
        return self.dropout(Quantize.apply(out2))


# ==============================================================================
# TRANSFORMER ENCODER LAYER
class TransformerEncoderLayer(nn.Module):
    def __init__(self, hidden_size=768, num_attention_heads=12,
                 intermediate_size=3072, dropout=0.1, layer_id=0):
        super().__init__()
        self.layer_id       = layer_id
        self.attention      = MultiHeadSelfAttention(hidden_size,
                                                     num_attention_heads,
                                                     dropout, layer_id=layer_id)
        self.attn_layer_norm = ApproximateLayerNorm(hidden_size, eps=1e-12)
        self.ffn             = FeedForward(hidden_size, intermediate_size,
                                           dropout, layer_id=layer_id)
        self.ffn_layer_norm  = ApproximateLayerNorm(hidden_size, eps=1e-12)

    def forward(self, x, attention_mask=None):
        p = f'bert.encoder.layers.{self.layer_id}'
        attn_output  = self.attention(x, attention_mask=attention_mask)

        # ── Residual Add 1: INT8 + INT8 → de-quantise → LayerNorm → INT8
        if scale_tracker.enabled:
            scale_tracker.record_activation_scale(f'{p}.residual1.x_scale',    x)
            scale_tracker.record_activation_scale(f'{p}.residual1.attn_scale', attn_output)
        norm_input_1 = Quantize.apply(x) + Quantize.apply(attn_output)
        ln1_out = self.attn_layer_norm(norm_input_1)
        if scale_tracker.enabled:
            scale_tracker.record_activation_scale(
                f'{p}.attn_layer_norm.output_scale', ln1_out)
        x = Quantize.apply(ln1_out)

        ffn_output   = self.ffn(x)

        # ── Residual Add 2: same pattern
        if scale_tracker.enabled:
            scale_tracker.record_activation_scale(f'{p}.residual2.x_scale',   x)
            scale_tracker.record_activation_scale(f'{p}.residual2.ffn_scale', ffn_output)
        norm_input_2 = Quantize.apply(x) + Quantize.apply(ffn_output)
        ln2_out = self.ffn_layer_norm(norm_input_2)
        if scale_tracker.enabled:
            scale_tracker.record_activation_scale(
                f'{p}.ffn_layer_norm.output_scale', ln2_out)
        x = Quantize.apply(ln2_out)
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
    Saves three artefacts:
      <file_path>                          — INT8 weights + INT8 biases + scales (.npz)
      <file_path>_input_scales.npz         — per-layer input scales  (median of calibration)
      <file_path>_activation_scales.npz   — ALL activation-node scales (median of calibration)
      <file_path>_scale_metadata.json     — full per-node statistics (mean/median/p95/min/max)

    Every recorded scale — weight, bias, input, activation — is printed in a
    structured report so nothing is silently dropped.
    """
    import json

    torch.cuda.empty_cache()
    quantized_state_dict = {}

    # ── Pull everything ScaleTracker collected ─────────────────────────────────
    scale_stats = scale_tracker.get_statistics()

    # Separate the three pools that live inside scale_stats:
    #   1. layer_input_scales  →  keys that are plain layer paths
    #      (e.g. 'bert.encoder.layers.0.attention.q')
    #   2. activation_scales   →  keys that end with a node tag
    #      (e.g. '...q.output_scale', '...residual1.x_scale', '...gcu.output_scale')
    #   3. gelu_input_range    →  single special entry with a different structure
    #
    # We distinguish (1) from (2) by checking for a '.' in the last path segment:
    #   layer paths end in a short identifier like 'q', 'out', 'dense_1', 'classifier'
    #   activation keys end in something like 'output_scale', 'x_scale', 'ffn_scale'
    ACTIVATION_SUFFIXES = (
        'output_scale', 'x_scale', 'attn_scale', 'ffn_scale',
        'ctx_scale', 'qkt_buffer.output_scale',
    )

    input_scale_stats  = {}   # layer_input_scales entries
    act_scale_stats    = {}   # activation_scales  entries

    for key, stat in scale_stats.items():
        if key == 'gelu_input_range':
            continue
        if not isinstance(stat, dict) or 'median' not in stat:
            continue
        last_segment = key.rsplit('.', 1)[-1]
        if any(key.endswith(sfx) for sfx in ACTIVATION_SUFFIXES):
            act_scale_stats[key] = stat
        else:
            input_scale_stats[key] = stat

    # ══════════════════════════════════════════════════════════════════════════
    # SECTION 1 — WEIGHT & BIAS QUANTIZATION
    # ══════════════════════════════════════════════════════════════════════════
    print("\n" + "=" * 70)
    print("SECTION 1 — WEIGHT & BIAS QUANTIZATION  (INT8)")
    print("=" * 70)
    print(f"  {'Parameter':<60} {'weight_scale':>14}  {'input_scale':>14}  {'bias_scale':>14}")
    print("  " + "-" * 106)

    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue

        param_data = param.detach().cpu().numpy()

        # ── Weights (2-D → INT8) ──────────────────────────────────────────────
        if 'weight' in name and param_data.ndim == 2:
            abs_max      = np.abs(param_data).max()
            weight_scale = abs_max / 127.0 if abs_max > 0 else 1.0

            q_weight = np.round(param_data / weight_scale).clip(-128, 127).astype(np.int8)
            quantized_state_dict[name]            = q_weight
            quantized_state_dict[f'{name}.scale'] = np.float32(weight_scale)

            # Input scale: median from ScaleTracker calibration batches
            layer_key = name.replace('.weight', '')
            if layer_key in input_scale_stats:
                input_scale = input_scale_stats[layer_key]['median']
                calibrated  = True
            else:
                input_scale = 1.0
                calibrated  = False
            quantized_state_dict[f'{name}.input_scale'] = np.float32(input_scale)

            tag = '' if calibrated else '  ← default (no calibration data)'
            print(f"  {name:<60} {weight_scale:>14.6e}  {input_scale:>14.6e}{tag}")

        # ── Biases (1-D → INT8, scale = input_scale × weight_scale) ──────────
        elif 'bias' in name and param_data.ndim == 1:
            weight_name      = name.replace('.bias', '.weight')
            weight_scale_key = f'{weight_name}.scale'
            input_scale_key  = f'{weight_name}.input_scale'

            if (weight_scale_key in quantized_state_dict and
                    input_scale_key in quantized_state_dict):
                weight_scale = float(quantized_state_dict[weight_scale_key])
                input_scale  = float(quantized_state_dict[input_scale_key])
                bias_scale   = input_scale * weight_scale

                bias_int8 = np.round(param_data / bias_scale).clip(
                    -128, 127).astype(np.int8)

                quantized_state_dict[name]            = bias_int8
                quantized_state_dict[f'{name}.scale'] = np.float32(bias_scale)
                print(f"  {name:<60} {'INT8':>14}  {'':>14}  {bias_scale:>14.6e}")
            else:
                quantized_state_dict[name] = param_data.astype(np.float32)
                print(f"  {name:<60} {'FP32 fallback':>14}")

        # ── Other parameters (embeddings ≥2-D, LayerNorm 1-D, etc.) ──────────
        else:
            if param_data.ndim >= 2:
                abs_max = np.abs(param_data).max()
                scale   = abs_max / 127.0 if abs_max > 0 else 1.0
                q_param = np.round(param_data / scale).clip(-128, 127).astype(np.int8)
                quantized_state_dict[name]            = q_param
                quantized_state_dict[f'{name}.scale'] = np.float32(scale)
            else:
                quantized_state_dict[name] = param_data.astype(np.float32)

    # Save weights .npz
    np.savez_compressed(file_path, **quantized_state_dict)
    print(f"\n  Saved weights → {file_path}")

    # ══════════════════════════════════════════════════════════════════════════
    # SECTION 2 — INPUT SCALES  (layer_input_scales from ScaleTracker)
    # ══════════════════════════════════════════════════════════════════════════
    print("\n" + "=" * 70)
    print("SECTION 2 — LAYER INPUT SCALES  (calibrated over eval batches)")
    print("=" * 70)
    print(f"  {'Layer':<60} {'median':>12} {'mean':>12} {'p95':>12} {'min':>12} {'max':>12}  samples")
    print("  " + "-" * 122)

    input_scales_npz = {}
    # Sort by layer name so encoder layers appear in order
    for layer_key in sorted(input_scale_stats.keys()):
        stat = input_scale_stats[layer_key]
        input_scales_npz[layer_key] = np.float32(stat['median'])
        print(f"  {layer_key:<60} "
              f"{stat['median']:>12.6e} "
              f"{stat['mean']:>12.6e} "
              f"{stat['p95']:>12.6e} "
              f"{stat['min']:>12.6e} "
              f"{stat['max']:>12.6e}  "
              f"{stat['samples']}")

    input_scales_file = file_path.replace('.npz', '_input_scales.npz')
    np.savez_compressed(input_scales_file, **input_scales_npz)
    print(f"\n  Saved {len(input_scales_npz)} input scales → {input_scales_file}")

    # ══════════════════════════════════════════════════════════════════════════
    # SECTION 3 — ACTIVATION SCALES  (ALL nodes, no filtering)
    # ══════════════════════════════════════════════════════════════════════════
    print("\n" + "=" * 70)
    print("SECTION 3 — ACTIVATION SCALES  (all Quantize.apply sites)")
    print("=" * 70)
    print(f"  {'Node':<68} {'median':>12} {'mean':>12} {'p95':>12} {'min':>12} {'max':>12}  samples")
    print("  " + "-" * 130)

    act_scales_npz = {}
    # Group by encoder layer for readable output
    prev_layer = None
    for node_key in sorted(act_scale_stats.keys()):
        stat = act_scale_stats[node_key]
        act_scales_npz[node_key] = np.float32(stat['median'])

        # Print a blank separator between different encoder layers
        layer_id = node_key.split('.')[3] if node_key.startswith('bert.encoder.layers.') else None
        if layer_id != prev_layer:
            if prev_layer is not None:
                print()
            prev_layer = layer_id

        print(f"  {node_key:<68} "
              f"{stat['median']:>12.6e} "
              f"{stat['mean']:>12.6e} "
              f"{stat['p95']:>12.6e} "
              f"{stat['min']:>12.6e} "
              f"{stat['max']:>12.6e}  "
              f"{stat['samples']}")

    act_scales_file = file_path.replace('.npz', '_activation_scales.npz')
    np.savez_compressed(act_scales_file, **act_scales_npz)
    print(f"\n  Saved {len(act_scales_npz)} activation scales → {act_scales_file}")

    # ══════════════════════════════════════════════════════════════════════════
    # SECTION 4 — GELU INPUT RANGE ANALYSIS
    # ══════════════════════════════════════════════════════════════════════════
    if 'gelu_input_range' in scale_stats:
        g = scale_stats['gelu_input_range']
        print("\n" + "=" * 70)
        print("SECTION 4 — GELU INPUT RANGE ANALYSIS  (Q10.22 format check)")
        print("=" * 70)
        print(f"  Measured from {g['samples']} calibration batches")
        print(f"  Overall range:          [{g['overall_min']:.6f},  {g['overall_max']:.6f}]")
        print(f"  Typical range (p05–p95):[{g['typical_min_p05']:.6f},  {g['typical_max_p95']:.6f}]")
        print(f"  Median range:           [{g['median_min']:.6f},  {g['median_max']:.6f}]")
        print(f"  Mean of means:          {g['mean_of_means']:.6f}")
        print(f"  Mean of stds:           {g['mean_of_stds']:.6f}")

        max_abs   = max(abs(g['overall_min']), abs(g['overall_max']))
        int_bits  = int(np.ceil(np.log2(max_abs + 1))) + 1
        frac_bits = 32 - int_bits
        print(f"\n  Actual Q10.22 format covers: [{-2**9:.1f}, {2**9 - 1:.1f}]  "
              f"precision = {2**-22:.8f}")
        print(f"  Data fits in Q{int_bits}.{frac_bits}  "
              f"(range [{-2**(int_bits-1):.1f}, {2**(int_bits-1)-1:.1f}],  "
              f"precision {2**-frac_bits:.8f})")

        range_span = g['typical_max_p95'] - g['typical_min_p05']
        print(f"\n  PLA segment size over typical range ({range_span:.4f}):")
        for n in [8, 16, 32]:
            print(f"    {n:2d} segments → {range_span/n:.6f} per segment  "
                  f"({n*4} bytes for K/B LUT)")

    # ══════════════════════════════════════════════════════════════════════════
    # SECTION 5 — FULL METADATA JSON  (all statistics for every node)
    # ══════════════════════════════════════════════════════════════════════════
    metadata_file = file_path.replace('.npz', '_scale_metadata.json')
    with open(metadata_file, 'w') as f:
        json.dump(scale_stats, f, indent=2)
    print(f"\n  Saved full scale metadata (all nodes, all stats) → {metadata_file}")

    # ══════════════════════════════════════════════════════════════════════════
    # FINAL SUMMARY
    # ══════════════════════════════════════════════════════════════════════════
    print("\n" + "=" * 70)
    print("EXTRACTION SUMMARY")
    print("=" * 70)
    total_mb      = sum(v.nbytes for v in quantized_state_dict.values()) / (1024 ** 2)
    n_int8_weights = sum(1 for k, v in quantized_state_dict.items()
                         if 'weight' in k and not k.endswith('.scale')
                         and hasattr(v, 'dtype') and v.dtype == np.int8)
    n_int8_bias    = sum(1 for k, v in quantized_state_dict.items()
                         if 'bias' in k and not k.endswith('.scale')
                         and hasattr(v, 'dtype') and v.dtype == np.int8)
    n_input_scales = len(input_scales_npz)
    n_act_scales   = len(act_scales_npz)
    print(f"  INT8 weight tensors saved:  {n_int8_weights}")
    print(f"  INT8 bias tensors saved:    {n_int8_bias}")
    print(f"  Layer input scales saved:   {n_input_scales}   → {input_scales_file}")
    print(f"  Activation scales saved:    {n_act_scales}   → {act_scales_file}")
    print(f"  Full metadata (JSON):                         → {metadata_file}")
    print(f"  Total weight file size:     {total_mb:.2f} MB")
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
    print("Quantization: PER-TENSOR (hardware-accurate)")
    print("Biases:       INT8  (bias_scale = input_scale × weight_scale)")
    print("GCU pipeline: EU1 → DU → EU2  |  input Q10.22  →  output Q48.16\n")

    optimizer            = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE)
    num_training_steps   = NUM_EPOCHS * len(train_dataloader)
    lr_scheduler         = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps   = int(0.1 * num_training_steps),
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
    output_weights_file = '/kaggle/working/weights_with_scales.npz'
    extract_and_save_quantized_weights(model, output_weights_file)


if __name__ == "__main__":
    main()