import math, sys
import numpy as np
from pathlib import Path


# ═══════════════════════════════════════════════════════════════════════════════
# QUANTIZE

def fake_quant(x: np.ndarray):
    """
    Per-tensor fake-quantization.
    scale = max(|x|) / 127
    return clip(round(x/scale), -127, 127) * scale  (stays float64)
    """
    x    = x.astype(np.float64)
    amax = float(np.abs(x).max())
    sc   = max(amax / 127.0, 1e-8)
    q    = np.clip(np.round(x / sc), -127.0, 127.0) * sc
    return q, np.float32(sc)


# ═══════════════════════════════════════════════════════════════════════════════
# FIXED-POINT

def to_fixed(x: np.ndarray, bits: int, frac_bits: int) -> np.ndarray:
    scale = 2.0 ** frac_bits
    lo    = -(2.0 ** (bits - 1))
    hi    =  (2.0 ** (bits - 1)) - 1.0
    return np.clip(np.round(x.astype(np.float64) * scale), lo, hi) / scale


# ═══════════════════════════════════════════════════════════════════════════════
# PLA SOFTMAX

def _build_pla():
    xs        = np.linspace(-10.0, 0.0, 1001)
    ys        = np.exp(xs)
    intervals = np.linspace(-10.0, 0.0, 13)
    ms, cs, as_ = [], [], []
    for i in range(12):
        a, b   = intervals[i], intervals[i + 1]
        mask   = (xs >= a) & (xs <= b)
        m, c   = np.polyfit(xs[mask], ys[mask], 1)
        ms.append(m); cs.append(c); as_.append(a)
    return (np.array(as_, np.float64),
            np.array(ms,  np.float64),
            np.array(cs,  np.float64))

_PLA_INTERVALS, _PLA_M, _PLA_C = _build_pla()

def _pla_exp(x: np.ndarray) -> np.ndarray:
    """Port of PLASoftmax.pla_exp()."""
    xc  = np.clip(x.astype(np.float64), -10.0, 0.0)
    idx = np.clip(
        np.sum(xc[..., np.newaxis] >= _PLA_INTERVALS, axis=-1) - 1,
        0, 11)
    return _PLA_M[idx] * xc + _PLA_C[idx]

def pla_softmax(scores: np.ndarray):
    """
    Port of PLASoftmax.forward():
      max-subtract → Q5.26 → pla_exp → / sum → fake_quant
    Input is already INT8-grid (quantized QKt scores).
    Returns (float64_on_int8_grid, scale).
    """
    mx      = scores.max(axis=-1, keepdims=True)
    shifted = to_fixed(scores - mx, 32, 26)       # Q5.26
    exps    = _pla_exp(shifted)
    probs   = exps / (exps.sum(axis=-1, keepdims=True) + 1e-9)
    softmax_q1_15 = to_fixed(probs, 16, 15)       # Q1.15 intermediate
    return fake_quant(softmax_q1_15)               # Q1.15 → INT8


# ═══════════════════════════════════════════════════════════════════════════════
# GCU — GELU COMPUTE UNIT
# Matches new training script exactly:
#   EU1, DU, EU2 three-stage pipeline
#   PolynomialUnit: s(x) = -2*log2(e)*sqrt(2/pi) * (x + 0.044715*x^3)
#   ExponentialUnit: precise log2(e), asymmetric barrel shifter
#   DivisionUnit: returns (exponent, sign); EU2 applied outside

# ── ExponentialUnit LUT ───────────────────────────────────────────────────────
_LOG2E = math.log2(math.e)   # 1.4426950408889634  (precise, matches new script)

def _build_exp_coeffs(num_segments=8):
    coeffs = []
    for i in range(num_segments):
        x_s, x_e = i / num_segments, (i + 1) / num_segments
        y_s, y_e = 2.0 ** x_s, 2.0 ** x_e
        K = (y_e - y_s) / (x_e - x_s)
        B = y_s - K * x_s
        coeffs.append((K, B))
    return np.array(coeffs, np.float64)   # (8, 2)

_EXP_COEFFS = _build_exp_coeffs()

def _exp_unit(x: np.ndarray, use_log2e: bool = True) -> np.ndarray:
    """
    Port of ExponentialUnit.forward() — new version.
    Asymmetric barrel shifter: positive int -> left shift (multiply),
                               negative int -> right shift (divide).
    Precise log2(e) = 1.4426950408889634.
    """
    x = x.astype(np.float64)
    if use_log2e:
        x = x * _LOG2E                             # precise log2(e)
    x_int  = np.floor(x).astype(np.int64)
    x_frac = np.clip(x - x_int.astype(np.float64), 0.0, 0.999999)
    seg    = np.clip(np.floor(x_frac * 8).astype(np.int64), 0, 7)
    K = _EXP_COEFFS[seg, 0];  B = _EXP_COEFFS[seg, 1]
    mantissa = K * x_frac + B

    # Asymmetric barrel shifter — matches new training script
    result = np.where(
        x_int >= 0,
        mantissa * (2.0 ** np.clip(x_int,  0, 15).astype(np.float64)),   # left shift
        mantissa / (2.0 ** np.clip(-x_int, 0, 15).astype(np.float64))    # right shift
    )
    return result


def _leading_one(x: np.ndarray):
    """Port of DivisionUnit.leading_one_detector()."""
    abs_x  = np.maximum(np.abs(x.astype(np.float64)), 1e-8)
    log2_x = np.log2(abs_x)
    w      = np.floor(log2_x).astype(np.int64)
    m      = log2_x - w.astype(np.float64) + 1.0
    return w, m

_DU_Q = 16   # fractional bits for Q48.16

def _div_unit(num: np.ndarray, den: np.ndarray, add_one: bool = False):
    """
    Port of DivisionUnit.forward() — new version.
    Returns (exponent, sign) tuple — EU2 is applied by the caller (GCU).
    s_i = (w_i - Q)  (not shifted, just float difference)
    exponent = (m1 + s1) - (m2 + s2)
    """
    if add_one:
        den = 1.0 + den
    w1, m1 = _leading_one(num)
    w2, m2 = _leading_one(den)
    s1 = w1.astype(np.float64) - _DU_Q
    s2 = w2.astype(np.float64) - _DU_Q
    exponent = (m1 + s1) - (m2 + s2)
    sign     = np.sign(num) * np.sign(den)
    return exponent, sign


def gcu(x: np.ndarray) -> np.ndarray:
    """
    Port of GCU.forward() — new three-stage pipeline:
      Stage 1: PolynomialUnit  → s(x)
      Stage 2: EU1             → exp(s(x))
      Stage 3: DU              → exponent, sign
      Stage 4: EU2             → result = EU2(exponent) * sign

    PolynomialUnit:
      s(x) = -2*log2(e)*sqrt(2/pi) * (x + 0.044715*x^3)
    """
    x = x.astype(np.float64)

    # Stage 1 — PolynomialUnit
    coeff = -2.0 * _LOG2E * math.sqrt(2.0 / math.pi)   # ≈ -2.3046
    cubic = 0.044715
    s_x   = coeff * (x + cubic * (x ** 3))

    # Stage 2 — EU1: exp(s_x), no log2e scaling (s_x already in log2 domain)
    exp_term = _exp_unit(s_x, use_log2e=False)

    # Stage 3 — DU: returns (exponent, sign)
    exponent, sign = _div_unit(x, exp_term, add_one=True)

    # Stage 4 — EU2: apply exponential to exponent
    result = _exp_unit(exponent, use_log2e=False) * sign
    return result


# ═══════════════════════════════════════════════════════════════════════════════
# APPROXIMATE LAYER NORM

def approx_layernorm(x: np.ndarray, gamma: np.ndarray, beta: np.ndarray,
                     eps: float = 1e-12, nr: int = 8) -> np.ndarray:
    """
    Port of ApproximateLayerNorm.forward():
      var → Q5.26 → Newton-Raphson sqrt (8 iterations) → normalise → affine
    """
    mean   = x.mean(axis=-1, keepdims=True)
    var    = x.var(axis=-1,  keepdims=True)
    var_fx = to_fixed(var, 32, 26)              # Q5.26

    s = np.where(var_fx > 1.0, var_fx * 0.5, np.ones_like(var_fx))
    for _ in range(nr):
        s = 0.5 * (s + var_fx / (s + 1e-9))

    x_norm = (x - mean) / (s + eps)
    return gamma.astype(np.float64) * x_norm + beta.astype(np.float64)


# ═══════════════════════════════════════════════════════════════════════════════
# QUANTIZED LINEAR

def quantized_linear(x: np.ndarray, W_f: np.ndarray,
                     b_f: np.ndarray = None) -> np.ndarray:
    """
    Port of QuantizedLinear.forward():
      fake_quant(input) @ fake_quant(weight).T + bias
    W_f already on INT8 grid; fake_quant is near no-op.
    b_f is plain float (INT32 * bias_scale).
    """
    x_fq,  _ = fake_quant(x)
    W_fq,  _ = fake_quant(W_f)
    out = x_fq @ W_fq.T.astype(np.float64)
    if b_f is not None:
        out = out + b_f.astype(np.float64)
    return out


# ═══════════════════════════════════════════════════════════════════════════════
# MULTI-HEAD SELF-ATTENTION

class AttentionBlock:
    TILE = 32

    def __init__(self, hidden=768, heads=12):
        self.hidden   = hidden
        self.heads    = heads
        self.head_dim = hidden // heads
        self.scale    = self.head_dim ** -0.5
        self.Wq = self.Wk = self.Wv = self.Wo = None
        self.bq = self.bk = self.bv = self.bo = None

    def _split(self, x, B, S):
        return x.reshape(B, S, self.heads, self.head_dim).transpose(0, 2, 1, 3)

    def forward(self, x: np.ndarray, mask=None):
        B, S, _ = x.shape
        T        = self.TILE

        # ── Q/K/V projections
        q_full = self._split(quantized_linear(x, self.Wq, self.bq), B, S)
        k_full = self._split(quantized_linear(x, self.Wk, self.bk), B, S)
        v_full = self._split(quantized_linear(x, self.Wv, self.bv), B, S)

        # ── QKt tiled
        H  = self.heads
        scores = np.zeros((B, H, S, S), np.float64)
        for i in range(0, S, T):
            for j in range(0, S, T):
                q_tile, _ = fake_quant(q_full[:, :, i:i+T, :])
                k_tile, _ = fake_quant(k_full[:, :, j:j+T, :])
                scores[:, :, i:i+T, j:j+T] = (
                    q_tile @ k_tile.transpose(0, 1, 3, 2)) * self.scale

        if mask is not None:
            scores = scores + mask

        # ── NEW: Quantize QKt scores to INT8 before softmax
        # Matches training: attn_scores_int8 = Quantize.apply(attn_scores)
        scores_fq, _ = fake_quant(scores)

        # ── PLASoftmax: receives INT8-grid scores
        attn_probs, _ = pla_softmax(scores_fq)

        # ── V: Quantize.apply once on entire v_full, then tiled matmul
        v_fq, _ = fake_quant(v_full)
        D       = self.head_dim
        ctx     = np.zeros((B, H, S, D), np.float64)
        for i in range(0, S, T):
            ctx[:, :, i:i+T, :] = attn_probs[:, :, i:i+T, :] @ v_fq

        # merge heads → Wo
        ctx = ctx.transpose(0, 2, 1, 3).reshape(B, S, self.hidden)
        return quantized_linear(ctx, self.Wo, self.bo)


# ═══════════════════════════════════════════════════════════════════════════════
# FEED-FORWARD

class FFNBlock:
    TILE = 32

    def __init__(self):
        self.W1 = self.W2 = None
        self.b1 = self.b2 = None

    def forward(self, x: np.ndarray):
        B, S, _  = x.shape
        T        = self.TILE

        # ── ffn1 GeMM
        out1 = np.zeros((B, S, self.W1.shape[0]), np.float64)
        for i in range(0, S, T):
            out1[:, i:i+T, :] = quantized_linear(x[:, i:i+T, :], self.W1, self.b1)

        # ── NEW: Quantize ffn1 output to INT8 intermediate buffer
        # Matches training: out1_int8 = Quantize.apply(out1)
        out1_int8, _ = fake_quant(out1)

        # ── Dequant INT8 buffer → Q10.22 → GCU → Q48.16 → INT8
        # Matches training: out1_q10_22 = to_fixed_point(out1_int8, 32, 22)
        out1_q10_22 = to_fixed(out1_int8, 32, 22)
        gcu_out     = gcu(out1_q10_22)
        gcu_q48_16  = to_fixed(gcu_out, 64, 16)                  # Q48.16
        x_mid, _    = fake_quant(gcu_q48_16)                     # Q48.16 → INT8

        # ── ffn2 GeMM
        out2 = np.zeros((B, S, self.W2.shape[0]), np.float64)
        for i in range(0, S, T):
            out2[:, i:i+T, :] = quantized_linear(x_mid[:, i:i+T, :], self.W2, self.b2)
        return out2


# ═══════════════════════════════════════════════════════════════════════════════
# TRANSFORMER ENCODER LAYER

class EncoderLayer:
    def __init__(self, hidden=768, heads=12):
        self.attn = AttentionBlock(hidden, heads)
        self.ffn  = FFNBlock()
        self.ln1_gamma = self.ln1_beta = None
        self.ln2_gamma = self.ln2_beta = None

    def forward(self, x: np.ndarray, mask=None):
        attn_out = self.attn.forward(x, mask)

        # Residual 1
        x_fq,    _ = fake_quant(x)
        attn_fq, _ = fake_quant(attn_out)
        res1        = x_fq + attn_fq
        res1_q5_26  = to_fixed(res1, 32, 26)          # Q5.26 before LN
        x = approx_layernorm(res1_q5_26, self.ln1_gamma, self.ln1_beta, eps=1e-12)
        x_fq2, _ = fake_quant(x)                      # LN1 → INT8
        x = x_fq2

        ffn_out = self.ffn.forward(x)

        # Residual 2
        x_fq3,   _ = fake_quant(x)
        ffn_fq,  _ = fake_quant(ffn_out)
        res2        = x_fq3 + ffn_fq
        res2_q5_26  = to_fixed(res2, 32, 26)          # Q5.26 before LN
        x = approx_layernorm(res2_q5_26, self.ln2_gamma, self.ln2_beta, eps=1e-12)
        x_fq4, _ = fake_quant(x)                      # LN2 → INT8
        return x_fq4


# ═══════════════════════════════════════════════════════════════════════════════
# FULL BERT MODEL

class HardwareBERT:
    def __init__(self, num_layers=12, hidden=768, heads=12):
        self.hidden  = hidden
        self.layers  = [EncoderLayer(hidden, heads) for _ in range(num_layers)]
        self.emb_ln_gamma = self.emb_ln_beta = None
        self.word_emb = self.pos_emb = self.tok_type_emb = None
        self.pooler_W = self.pooler_b = None
        self.clf_W    = self.clf_b    = None

    def load_weights(self, npz_path: str):
        print(f"Loading: {npz_path}")
        W = np.load(npz_path, allow_pickle=True)

        def get(k, default=None):
            return W[k] if k in W.files else default

        def w_scale(k):
            v = get(f'{k}.scale');  return float(v) if v is not None else 1.0

        def b_scale(k):
            v = get(f'{k}.scale');  return float(v) if v is not None else 1.0

        def load_weight_f(k):
            raw = get(k)
            if raw is None: return None
            sc = w_scale(k)
            return (raw.astype(np.float32) * sc if raw.dtype == np.int8
                    else raw.astype(np.float32))

        def load_bias_f(k):
            raw = get(k)
            if raw is None: return None
            sc = b_scale(k)
            return (raw.astype(np.float64) * sc if raw.dtype == np.int32
                    else raw.astype(np.float64))

        def load_emb_f(k):
            raw = get(k)
            if raw is None: return None
            sc = w_scale(k)
            return (raw.astype(np.float32) * sc if raw.dtype == np.int8
                    else raw.astype(np.float32))

        # Embeddings
        self.word_emb     = load_emb_f('bert.embeddings.word_embeddings.weight')
        self.pos_emb      = load_emb_f('bert.embeddings.position_embeddings.weight')
        self.tok_type_emb = load_emb_f('bert.embeddings.token_type_embeddings.weight')
        self.emb_ln_gamma = get('bert.embeddings.LayerNorm.weight',
                                np.ones(self.hidden,  np.float32))
        self.emb_ln_beta  = get('bert.embeddings.LayerNorm.bias',
                                np.zeros(self.hidden, np.float32))

        # Encoder layers
        for i, layer in enumerate(self.layers):
            pa = f'bert.encoder.layers.{i}.attention'
            pf = f'bert.encoder.layers.{i}.ffn'
            a  = layer.attn
            ff = layer.ffn

            a.Wq = load_weight_f(f'{pa}.q.weight')
            a.Wk = load_weight_f(f'{pa}.k.weight')
            a.Wv = load_weight_f(f'{pa}.v.weight')
            a.Wo = load_weight_f(f'{pa}.out.weight')

            a.bq = load_bias_f(f'{pa}.q.bias')
            a.bk = load_bias_f(f'{pa}.k.bias')
            a.bv = load_bias_f(f'{pa}.v.bias')
            a.bo = load_bias_f(f'{pa}.out.bias')

            ff.W1 = load_weight_f(f'{pf}.dense_1.weight')
            ff.W2 = load_weight_f(f'{pf}.dense_2.weight')
            ff.b1 = load_bias_f(f'{pf}.dense_1.bias')
            ff.b2 = load_bias_f(f'{pf}.dense_2.bias')

            layer.ln1_gamma = get(f'bert.encoder.layers.{i}.attn_layer_norm.weight',
                                  np.ones(self.hidden,  np.float32))
            layer.ln1_beta  = get(f'bert.encoder.layers.{i}.attn_layer_norm.bias',
                                  np.zeros(self.hidden, np.float32))
            layer.ln2_gamma = get(f'bert.encoder.layers.{i}.ffn_layer_norm.weight',
                                  np.ones(self.hidden,  np.float32))
            layer.ln2_beta  = get(f'bert.encoder.layers.{i}.ffn_layer_norm.bias',
                                  np.zeros(self.hidden, np.float32))

        # Pooler (plain float)
        self.pooler_W = load_weight_f('bert.pooler.weight')
        self.pooler_b = load_bias_f('bert.pooler.bias')

        # Classifier
        self.clf_W = load_weight_f('classifier.weight')
        self.clf_b = load_bias_f('classifier.bias')

        print(f"  Loaded {len(W.files)} tensors.")

    def forward(self, input_ids, token_type_ids, attention_mask):
        B, S = input_ids.shape
        pos  = np.arange(S)[np.newaxis, :]

        emb = (self.word_emb[input_ids].astype(np.float64)
             + self.pos_emb[pos].astype(np.float64)
             + self.tok_type_emb[token_type_ids].astype(np.float64))
        x   = approx_layernorm(emb, self.emb_ln_gamma, self.emb_ln_beta, eps=1e-12)

        mask = (1.0 - attention_mask[:, np.newaxis, np.newaxis, :]) * -10000.0

        for layer in self.layers:
            x = layer.forward(x, mask)

        # Pooler: CLS token → float linear → Tanh
        cls_f  = x[:, 0, :].astype(np.float64)
        pooled = np.tanh(
            cls_f @ self.pooler_W.T.astype(np.float64) + self.pooler_b.astype(np.float64))

        # Classifier
        logits = quantized_linear(pooled, self.clf_W, self.clf_b)
        return logits.astype(np.float32)

    def predict(self, input_ids, token_type_ids, attention_mask):
        return np.argmax(self.forward(input_ids, token_type_ids, attention_mask), axis=-1)


# ═══════════════════════════════════════════════════════════════════════════════
# ACCURACY ESTIMATION

def estimate_accuracy(npz_path: str, num_samples: int = 872):
    try:
        from transformers import BertTokenizer
        from datasets import load_dataset
    except ImportError:
        print("pip install transformers datasets"); return None

    print("\n" + "="*60)
    print("  HARDWARE-FAITHFUL INT8 BERT  --  SST-2 VALIDATION")
    print("  New GCU (EU1+DU+EU2) | INT8 QKt buffer | INT8 FFN1 buffer")
    print("="*60)

    model = HardwareBERT()
    model.load_weights(npz_path)

    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    dataset   = load_dataset("glue", "sst2", split="validation")
    samples   = dataset.select(range(min(num_samples, len(dataset))))
    correct = total = 0

    print(f"\n  Evaluating {len(samples)} samples...\n")
    for start in range(0, len(samples), 8):
        batch  = samples[start:start+8]
        labels = np.array(batch['label'])
        enc    = tokenizer(batch['sentence'], padding='max_length',
                           truncation=True, max_length=128, return_tensors='np')
        preds  = model.predict(enc['input_ids'].astype(np.int64),
                               enc['token_type_ids'].astype(np.int64),
                               enc['attention_mask'].astype(np.float32))
        correct += int((preds == labels).sum())
        total   += len(labels)
        print(f"  [{total:>4}/{len(samples)}]  {correct/total*100:.2f}%", end='\r')

    acc = correct / total * 100
    print(f"\n\n  Accuracy  : {acc:.2f}%  ({correct}/{total})")
    print(f"  Expected  : ~80-88%")
    print("="*60)
    return acc




# ═══════════════════════════════════════════════════════════════════════════════
# ACTIVATION RECORDER
#
# Records full activation matrices before/after every PDF block.
# Updated for new training script:
#   - QKt scores quantized to INT8 buffer before Q5.26 snap + softmax
#   - FFN1 output quantized to INT8 buffer before Q10.22 + GCU
#   - LN inputs snapped to Q5.26 before LayerNorm (matches PDF hardware flow)
#   - GCU uses new EU1->DU->EU2 pipeline (via updated gcu() function)
#
# Saved dtypes:
#   INT8   — any node output after quantize  (+ companion _scale float32 key)
#   INT32  — GeMM accumulators and after-bias
#   float32— fixed-point registers and LN domain values

def record_activations(npz_path: str,
                       out_npz:   str = "activations.npz",
                       out_index: str = "activations_index.txt",
                       n_samples: int = 5):
    """
    Runs n_samples through the full 12-layer model and records every
    activation matrix before and after each PDF block.
    """
    try:
        from transformers import BertTokenizer
        from datasets import load_dataset
    except ImportError:
        print("pip install transformers datasets"); return

    print("\n" + "="*60)
    print("  ACTIVATION RECORDER  (hardware-accurate dtypes)")
    print(f"  Samples: {n_samples}  |  Layers: 12  |  INT8/INT32/float32")
    print("="*60)

    model = HardwareBERT()
    model.load_weights(npz_path)

    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    dataset   = load_dataset("glue", "sst2", split="validation")

    store = {}
    index = {}

    def to_int8_arr(fq_arr, sc):
        return np.round(np.array(fq_arr) / float(sc)).clip(-127, 127).astype(np.int8)

    def int8_matmul_i32(a_i8, b_i8):
        return a_i8.astype(np.int32) @ b_i8.T.astype(np.int32)

    def save_i8(key, fq_arr, sc, desc):
        arr = to_int8_arr(fq_arr, sc)
        if arr.ndim > 1 and arr.shape[0] == 1:
            arr = arr.squeeze(0)
        store[key]            = arr
        store[key + "_scale"] = np.float32(sc)
        index[key]            = f"INT8    | {desc}"
        index[key + "_scale"] = f"float32 | scale for {key}"

    def save_i32(key, acc_i32, desc):
        arr = np.array(acc_i32).astype(np.int32)
        if arr.ndim > 1 and arr.shape[0] == 1:
            arr = arr.squeeze(0)
        store[key] = arr
        index[key] = f"INT32   | {desc}"

    def save_f32(key, arr, desc):
        a = np.array(arr).astype(np.float32)
        if a.ndim > 1 and a.shape[0] == 1:
            a = a.squeeze(0)
        store[key] = a
        index[key] = f"float32 | {desc}"

    T = 32

    for si in range(n_samples):
        sample = dataset[si]
        enc    = tokenizer(sample['sentence'], padding='max_length',
                           truncation=True, max_length=128, return_tensors='np')
        input_ids      = enc['input_ids'].astype(np.int64)
        token_type_ids = enc['token_type_ids'].astype(np.int64)
        attention_mask = enc['attention_mask'].astype(np.float32)

        print(f"\n  Sample {si}: \"{sample['sentence'][:70]}\"")
        print(f"  Label  : {sample['label']} ({'pos' if sample['label'] else 'neg'})")

        B, S = input_ids.shape
        pos  = np.arange(S)[np.newaxis, :]
        emb  = (model.word_emb[input_ids].astype(np.float64)
              + model.pos_emb[pos].astype(np.float64)
              + model.tok_type_emb[token_type_ids].astype(np.float64))
        x    = approx_layernorm(emb, model.emb_ln_gamma, model.emb_ln_beta, eps=1e-12)
        mask = (1.0 - attention_mask[:, np.newaxis, np.newaxis, :]) * -10000.0

        for li in range(len(model.layers)):
            p     = f"s{si}_l{li}"
            layer = model.layers[li]
            a     = layer.attn
            ff    = layer.ffn
            H, D  = a.heads, a.head_dim

            # ── INPUT QUANTIZE ────────────────────────────────────────────
            save_f32(f"{p}_input_quant_before", x[0],
                f"s{si} l{li} | INPUT_QUANT before | float LN output entering quantizer | ({S},{x.shape[-1]})")
            x_fq, x_sc = fake_quant(x)
            save_i8(f"{p}_input_quant_after", x_fq, x_sc,
                f"s{si} l{li} | INPUT_QUANT after  | x_i8 | ({S},{x_fq.shape[-1]})")

            x_i8_raw  = to_int8_arr(x_fq,  x_sc)
            Wq_i8_raw = to_int8_arr(a.Wq,  fake_quant(a.Wq)[1])
            Wk_i8_raw = to_int8_arr(a.Wk,  fake_quant(a.Wk)[1])
            Wv_i8_raw = to_int8_arr(a.Wv,  fake_quant(a.Wv)[1])
            Wo_i8_raw = to_int8_arr(a.Wo,  fake_quant(a.Wo)[1])

            # ── Q PROJECTION ──────────────────────────────────────────────
            q_acc_i32 = int8_matmul_i32(x_i8_raw[0], Wq_i8_raw)
            save_i32(f"{p}_q_proj_after_gemm", q_acc_i32,
                f"s{si} l{li} | Q_PROJ after_gemm | INT32 = x_i8 @ Wq.T | ({S},{q_acc_i32.shape[-1]})")
            if a.bq is not None:
                bq_sc      = float(x_sc) * float(fake_quant(a.Wq)[1])
                bq_i32     = np.round(a.bq / bq_sc).clip(-2147483648, 2147483647).astype(np.int32)
                q_bias_i32 = q_acc_i32 + bq_i32
                save_i32(f"{p}_q_proj_after_bias", q_bias_i32,
                    f"s{si} l{li} | Q_PROJ after_bias  | INT32 + bias_i32 | ({S},{q_bias_i32.shape[-1]})")
            else:
                q_bias_i32 = q_acc_i32
            q_sc_combined = float(x_sc) * float(fake_quant(a.Wq)[1])
            q_dequant     = q_bias_i32.astype(np.float64) * q_sc_combined
            q_fq, q_sc    = fake_quant(q_dequant[np.newaxis])
            save_i8(f"{p}_q_proj_after_quant", q_fq, q_sc,
                f"s{si} l{li} | Q_PROJ after_quant | q_i8 [32int->8] | ({S},{q_fq.shape[-1]})")

            # ── K PROJECTION ──────────────────────────────────────────────
            k_acc_i32 = int8_matmul_i32(x_i8_raw[0], Wk_i8_raw)
            save_i32(f"{p}_k_proj_after_gemm", k_acc_i32,
                f"s{si} l{li} | K_PROJ after_gemm | INT32 = x_i8 @ Wk.T | ({S},{k_acc_i32.shape[-1]})")
            if a.bk is not None:
                bk_sc      = float(x_sc) * float(fake_quant(a.Wk)[1])
                bk_i32     = np.round(a.bk / bk_sc).clip(-2147483648, 2147483647).astype(np.int32)
                k_bias_i32 = k_acc_i32 + bk_i32
                save_i32(f"{p}_k_proj_after_bias", k_bias_i32,
                    f"s{si} l{li} | K_PROJ after_bias  | INT32 + bias_i32 | ({S},{k_bias_i32.shape[-1]})")
            else:
                k_bias_i32 = k_acc_i32
            k_sc_combined = float(x_sc) * float(fake_quant(a.Wk)[1])
            k_dequant     = k_bias_i32.astype(np.float64) * k_sc_combined
            k_fq, k_sc    = fake_quant(k_dequant[np.newaxis])
            save_i8(f"{p}_k_proj_after_quant", k_fq, k_sc,
                f"s{si} l{li} | K_PROJ after_quant | k_i8 [32int->8] | ({S},{k_fq.shape[-1]})")

            # ── V PROJECTION ──────────────────────────────────────────────
            v_acc_i32 = int8_matmul_i32(x_i8_raw[0], Wv_i8_raw)
            save_i32(f"{p}_v_proj_after_gemm", v_acc_i32,
                f"s{si} l{li} | V_PROJ after_gemm | INT32 = x_i8 @ Wv.T | ({S},{v_acc_i32.shape[-1]})")
            if a.bv is not None:
                bv_sc      = float(x_sc) * float(fake_quant(a.Wv)[1])
                bv_i32     = np.round(a.bv / bv_sc).clip(-2147483648, 2147483647).astype(np.int32)
                v_bias_i32 = v_acc_i32 + bv_i32
                save_i32(f"{p}_v_proj_after_bias", v_bias_i32,
                    f"s{si} l{li} | V_PROJ after_bias  | INT32 + bias_i32 | ({S},{v_bias_i32.shape[-1]})")
            else:
                v_bias_i32 = v_acc_i32
            v_sc_combined = float(x_sc) * float(fake_quant(a.Wv)[1])
            v_dequant     = v_bias_i32.astype(np.float64) * v_sc_combined
            v_fq, v_sc_q  = fake_quant(v_dequant[np.newaxis])
            save_i8(f"{p}_v_proj_after_quant", v_fq, v_sc_q,
                f"s{si} l{li} | V_PROJ after_quant | v_i8 [32int->8] | ({S},{v_fq.shape[-1]})")

            # reshape to (H,S,D) for attention
            q_i8_h = to_int8_arr(q_fq, q_sc)[0].reshape(S, H, D).transpose(1, 0, 2)
            k_i8_h = to_int8_arr(k_fq, k_sc)[0].reshape(S, H, D).transpose(1, 0, 2)
            v_i8_h = to_int8_arr(v_fq, v_sc_q)[0].reshape(S, H, D).transpose(1, 0, 2)

            # ── QKt GeMM ─────────────────────────────────────────────────
            scores_i32 = np.zeros((H, S, S), np.int32)
            for h in range(H):
                scores_i32[h] = int8_matmul_i32(q_i8_h[h], k_i8_h[h])
            save_i32(f"{p}_qkt_after_gemm", scores_i32,
                f"s{si} l{li} | QKT_GEMM after_gemm | INT32 = q_i8 @ k_i8.T | ({H},{S},{S})")

            qkt_sc   = float(q_sc) * float(k_sc)
            scores_f = scores_i32.astype(np.float64) * qkt_sc * a.scale
            if mask is not None:
                scores_f = scores_f + mask[0]

            # NEW: quantize scores to INT8 buffer before softmax
            scores_fq, scores_buf_sc = fake_quant(scores_f)
            save_i8(f"{p}_qkt_after_quant", scores_fq, scores_buf_sc,
                f"s{si} l{li} | QKT_GEMM after_quant | INT8 buffer before softmax | ({H},{S},{S})")

            # dequant INT8 buffer → Q5.26 → softmax input
            mx      = scores_fq.max(axis=-1, keepdims=True)
            shifted = to_fixed(scores_fq - mx, 32, 26)
            save_f32(f"{p}_qkt_after_q5_26", shifted,
                f"s{si} l{li} | QKT_GEMM after_q5_26 | Q5.26 softmax input | ({H},{S},{S})")

            # ── SOFTMAX ───────────────────────────────────────────────────
            exps      = _pla_exp(shifted)
            probs_raw = exps / (exps.sum(axis=-1, keepdims=True) + 1e-9)
            save_f32(f"{p}_softmax_after_pla", probs_raw,
                f"s{si} l{li} | SOFTMAX after_pla   | TRUE SOFTMAX OUTPUT float32 | ({H},{S},{S})")
            attn_fq, attn_sc = fake_quant(probs_raw[np.newaxis])
            save_i8(f"{p}_softmax_after_quant", attn_fq, attn_sc,
                f"s{si} l{li} | SOFTMAX after_quant | attn_i8 [Q1.15->8] | ({H},{S},{S})")

            # ── .V GeMM ───────────────────────────────────────────────────
            attn_i8_raw = to_int8_arr(attn_fq, attn_sc)
            ctx_i32 = np.zeros((H, S, D), np.int32)
            for h in range(H):
                ctx_i32[h] = attn_i8_raw[0, h].astype(np.int32) @ v_i8_h[h].astype(np.int32)
            ctx_merged_i32 = ctx_i32.transpose(1, 0, 2).reshape(S, H * D)
            save_i32(f"{p}_dotv_after_gemm", ctx_merged_i32,
                f"s{si} l{li} | DOTV_GEMM after_gemm | INT32 = attn_i8 @ v_i8 | ({S},{H*D})")
            ctx_sc_combined = float(attn_sc) * float(v_sc_q)
            ctx_dequant     = ctx_merged_i32.astype(np.float64) * ctx_sc_combined
            ctx_fq, ctx_sc  = fake_quant(ctx_dequant[np.newaxis])
            save_i8(f"{p}_dotv_after_quant", ctx_fq, ctx_sc,
                f"s{si} l{li} | DOTV_GEMM after_quant | ctx_i8 [32int->8] | ({S},{H*D})")

            # ── Wo PROJECTION ─────────────────────────────────────────────
            ctx_i8_raw = to_int8_arr(ctx_fq, ctx_sc)[0]
            wo_acc_i32 = int8_matmul_i32(ctx_i8_raw, Wo_i8_raw)
            save_i32(f"{p}_wo_proj_after_gemm", wo_acc_i32,
                f"s{si} l{li} | WO_PROJ after_gemm | INT32 = ctx_i8 @ Wo.T | ({S},{wo_acc_i32.shape[-1]})")
            if a.bo is not None:
                bo_sc      = float(ctx_sc) * float(fake_quant(a.Wo)[1])
                bo_i32     = np.round(a.bo / bo_sc).clip(-2147483648, 2147483647).astype(np.int32)
                wo_bias_i32 = wo_acc_i32 + bo_i32
                save_i32(f"{p}_wo_proj_after_bias", wo_bias_i32,
                    f"s{si} l{li} | WO_PROJ after_bias  | INT32 + bias_i32 | ({S},{wo_bias_i32.shape[-1]})")
            else:
                wo_bias_i32 = wo_acc_i32
            wo_sc_combined = float(ctx_sc) * float(fake_quant(a.Wo)[1])
            wo_dequant     = wo_bias_i32.astype(np.float64) * wo_sc_combined
            wo_fq, wo_sc   = fake_quant(wo_dequant[np.newaxis])
            save_i8(f"{p}_wo_proj_after_quant", wo_fq, wo_sc,
                f"s{si} l{li} | WO_PROJ after_quant | wo_i8 [32int->8] | ({S},{wo_fq.shape[-1]})")

            # ── RESIDUAL ADD 1 + LN1 ──────────────────────────────────────
            x_res_fq,  x_res_sc  = fake_quant(x)
            wo_res_fq, wo_res_sc = fake_quant(wo_dequant[np.newaxis])
            x_res_i8  = to_int8_arr(x_res_fq,  x_res_sc)
            wo_res_i8 = to_int8_arr(wo_res_fq, wo_res_sc)
            res1_f    = (x_res_i8.astype(np.float64) * float(x_res_sc) +
                         wo_res_i8.astype(np.float64) * float(wo_res_sc))
            # Q5.26 snap before LN (matches new training script)
            res1_q5_26 = to_fixed(res1_f, 32, 26)
            save_f32(f"{p}_resadd1_ln1_input", res1_q5_26,
                f"s{si} l{li} | RESADD1_LN1 input | TRUE LN1 INPUT Q5.26 (x_i8*sc + wo_i8*sc) | ({S},{res1_q5_26.shape[-1]})")
            ln1_out = approx_layernorm(res1_q5_26, layer.ln1_gamma, layer.ln1_beta, eps=1e-12)
            save_f32(f"{p}_resadd1_ln1_after_ln", ln1_out,
                f"s{si} l{li} | RESADD1_LN1 after_ln | TRUE LN1 OUTPUT float32 | ({S},{ln1_out.shape[-1]})")
            x1_fq, x1_sc = fake_quant(ln1_out)
            save_i8(f"{p}_resadd1_ln1_after_quant", x1_fq, x1_sc,
                f"s{si} l{li} | RESADD1_LN1 after_quant | x_i8 post-LN1 | ({S},{x1_fq.shape[-1]})")

            # ── FFN1 GeMM ─────────────────────────────────────────────────
            x1_i8_raw    = to_int8_arr(x1_fq, x1_sc)[0]
            W1_fq, W1_sc = fake_quant(ff.W1)
            W1_i8_raw    = to_int8_arr(W1_fq, W1_sc)
            ffn1_acc_i32 = int8_matmul_i32(x1_i8_raw, W1_i8_raw)
            save_i32(f"{p}_ffn1_gemm_after_gemm", ffn1_acc_i32,
                f"s{si} l{li} | FFN1_GEMM after_gemm | INT32 = x_i8 @ W1.T | ({S},{ffn1_acc_i32.shape[-1]})")
            if ff.b1 is not None:
                b1_sc_combined = float(x1_sc) * float(W1_sc)
                b1_i32         = np.round(ff.b1 / b1_sc_combined).clip(-2147483648, 2147483647).astype(np.int32)
                ffn1_bias_i32  = ffn1_acc_i32 + b1_i32
                save_i32(f"{p}_ffn1_gemm_after_bias", ffn1_bias_i32,
                    f"s{si} l{li} | FFN1_GEMM after_bias | INT32 + bias_i32 | ({S},{ffn1_bias_i32.shape[-1]})")
            else:
                ffn1_bias_i32 = ffn1_acc_i32
            ffn1_sc_combined = float(x1_sc) * float(W1_sc)
            ffn1_dequant     = ffn1_bias_i32.astype(np.float64) * ffn1_sc_combined

            # NEW: INT8 intermediate buffer after ffn1 before GCU
            ffn1_buf_fq, ffn1_buf_sc = fake_quant(ffn1_dequant[np.newaxis])
            save_i8(f"{p}_ffn1_gemm_after_quant", ffn1_buf_fq, ffn1_buf_sc,
                f"s{si} l{li} | FFN1_GEMM after_quant | INT8 buffer before GCU | ({S},{ffn1_buf_fq.shape[-1]})")

            # dequant INT8 buffer → Q10.22 → GCU input
            ffn1_q10_22 = to_fixed(ffn1_buf_fq, 32, 22)
            save_f32(f"{p}_ffn1_gemm_after_q10_22", ffn1_q10_22[0] if ffn1_q10_22.ndim > 2 else ffn1_q10_22,
                f"s{si} l{li} | FFN1_GEMM after_q10_22 | Q10.22 GCU input | ({S},{ffn1_q10_22.shape[-1]})")

            # ── GCU (new EU1->DU->EU2 pipeline via updated gcu()) ─────────
            gcu_out    = gcu(ffn1_q10_22)
            gcu_q48_16 = to_fixed(gcu_out, 64, 16)
            save_f32(f"{p}_gcu_after_q48_16", gcu_q48_16[0] if gcu_q48_16.ndim > 2 else gcu_q48_16,
                f"s{si} l{li} | GCU after_q48_16 | TRUE GCU OUTPUT Q48.16 | ({S},{gcu_q48_16.shape[-1]})")
            mid_fq, mid_sc = fake_quant(gcu_q48_16)
            save_i8(f"{p}_gcu_after_quant", mid_fq, mid_sc,
                f"s{si} l{li} | GCU after_quant  | mid_i8 [Q48.16->8] | ({S},{mid_fq.shape[-1]})")

            # ── FFN2 GeMM ─────────────────────────────────────────────────
            mid_i8_raw   = to_int8_arr(mid_fq, mid_sc)[0]
            W2_fq, W2_sc = fake_quant(ff.W2)
            W2_i8_raw    = to_int8_arr(W2_fq, W2_sc)
            ffn2_acc_i32 = int8_matmul_i32(mid_i8_raw, W2_i8_raw)
            save_i32(f"{p}_ffn2_gemm_after_gemm", ffn2_acc_i32,
                f"s{si} l{li} | FFN2_GEMM after_gemm | INT32 = mid_i8 @ W2.T | ({S},{ffn2_acc_i32.shape[-1]})")
            if ff.b2 is not None:
                b2_sc_combined = float(mid_sc) * float(W2_sc)
                b2_i32         = np.round(ff.b2 / b2_sc_combined).clip(-2147483648, 2147483647).astype(np.int32)
                ffn2_bias_i32  = ffn2_acc_i32 + b2_i32
                save_i32(f"{p}_ffn2_gemm_after_bias", ffn2_bias_i32,
                    f"s{si} l{li} | FFN2_GEMM after_bias | INT32 + bias_i32 | ({S},{ffn2_bias_i32.shape[-1]})")
            else:
                ffn2_bias_i32 = ffn2_acc_i32
            ffn2_sc_combined = float(mid_sc) * float(W2_sc)
            ffn2_dequant     = ffn2_bias_i32.astype(np.float64) * ffn2_sc_combined
            ffn2_fq, ffn2_sc = fake_quant(ffn2_dequant[np.newaxis])
            save_i8(f"{p}_ffn2_gemm_after_quant", ffn2_fq, ffn2_sc,
                f"s{si} l{li} | FFN2_GEMM after_quant | ffn2_i8 [32int->8] | ({S},{ffn2_fq.shape[-1]})")

            # ── RESIDUAL ADD 2 + LN2 ──────────────────────────────────────
            x1_res_fq2,  x1_res_sc2  = fake_quant(ln1_out)
            ffn2_res_fq, ffn2_res_sc = fake_quant(ffn2_dequant[np.newaxis])
            x1_res_i8   = to_int8_arr(x1_res_fq2,  x1_res_sc2)
            ffn2_res_i8 = to_int8_arr(ffn2_res_fq, ffn2_res_sc)
            res2_f      = (x1_res_i8.astype(np.float64)  * float(x1_res_sc2) +
                           ffn2_res_i8.astype(np.float64) * float(ffn2_res_sc))
            # Q5.26 snap before LN (matches new training script)
            res2_q5_26  = to_fixed(res2_f, 32, 26)
            save_f32(f"{p}_resadd2_ln2_input", res2_q5_26,
                f"s{si} l{li} | RESADD2_LN2 input | TRUE LN2 INPUT Q5.26 (x_i8*sc + ffn2_i8*sc) | ({S},{res2_q5_26.shape[-1]})")
            ln2_out = approx_layernorm(res2_q5_26, layer.ln2_gamma, layer.ln2_beta, eps=1e-12)
            save_f32(f"{p}_resadd2_ln2_after_ln", ln2_out,
                f"s{si} l{li} | RESADD2_LN2 after_ln | TRUE LN2 OUTPUT float32 | ({S},{ln2_out.shape[-1]})")
            x2_fq, x2_sc = fake_quant(ln2_out)
            save_i8(f"{p}_resadd2_ln2_after_quant", x2_fq, x2_sc,
                f"s{si} l{li} | RESADD2_LN2 after_quant | x_i8 post-LN2 = LAYER OUTPUT | ({S},{x2_fq.shape[-1]})")

            x = ln2_out
            n_saved = len([k for k in store if k.startswith(f"s{si}_l{li}_")])
            print(f"    layer {li:2d} done  ({n_saved} tensors)")

    # ── save npz ──────────────────────────────────────────────────────────────
    print(f"\n  Saving {len(store)} tensors -> {out_npz} ...")
    np.savez_compressed(out_npz, **store)
    total_mb = sum(v.nbytes for v in store.values()) / 1024**2
    print(f"  Uncompressed : {total_mb:.1f} MB")

    # ── save index ────────────────────────────────────────────────────────────
    with open(out_index, 'w', encoding='utf-8') as f:
        f.write("=" * 78 + "\n")
        f.write("  ACTIVATION RECORDER - INDEX\n")
        f.write(f"  Tensors : {len(index)}   Samples : {n_samples}   Layers : 12\n")
        f.write("  INT8  keys have a companion _scale key (float32)\n")
        f.write("  INT32 keys are raw GeMM accumulators\n")
        f.write("  float32 keys are dequant/fixed-point domain values\n")
        f.write("=" * 78 + "\n\n")
        f.write("  LOAD:\n")
        f.write("    data    = np.load('activations.npz')\n")
        f.write("    q_i8    = data['s0_l0_q_proj_after_quant']        # int8\n")
        f.write("    q_scale = data['s0_l0_q_proj_after_quant_scale']  # float32\n")
        f.write("    q_float = q_i8.astype(np.float32) * q_scale\n\n")
        f.write("-" * 78 + "\n")
        prev_s, prev_l = None, None
        for key, desc in sorted(index.items()):
            parts  = key.split('_')
            si_tag = parts[0]
            li_tag = parts[1] if len(parts) > 1 else 'NA'
            if si_tag != prev_s:
                f.write(f"\n{'='*78}\n  {si_tag.upper()}\n{'='*78}\n")
                prev_s = si_tag; prev_l = None
            if li_tag != prev_l and li_tag.startswith('l'):
                f.write(f"\n  [{li_tag}]\n")
                prev_l = li_tag
            shape = list(store[key].shape)
            dtype = str(store[key].dtype)
            f.write(f"  {key:<60} {dtype:<8} shape={shape}\n")
            f.write(f"    {desc}\n\n")

    print(f"  Index saved  -> {out_index}")
    print(f"\n  Usage:")
    print(f"    data    = np.load('{out_npz}')")
    print(f"    q_i8    = data['s0_l0_q_proj_after_quant']")
    print(f"    q_scale = data['s0_l0_q_proj_after_quant_scale']")
    print("="*60)


if __name__ == "__main__":
    w = r"/kaggle/input/datasets/khalednabil676/bert-weight-file/weights_with_scales.npz"
    estimate_accuracy(w)