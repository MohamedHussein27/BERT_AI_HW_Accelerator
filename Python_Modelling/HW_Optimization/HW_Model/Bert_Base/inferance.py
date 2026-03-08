import math, sys
import numpy as np
from pathlib import Path


# ═══════════════════════════════════════════════════════════════════════════════
# QUANTIZE

def fake_quant(x: np.ndarray):
    """
    Per-tensor fake-quantization.
    Port of Quantize.forward():
      scale = max(|x|) / 127
      return clip(round(x/scale), -127, 127) * scale   ← stays float64
    Returns (quantized_float64, float32_scale).
    """
    x    = x.astype(np.float64)
    amax = float(np.abs(x).max())
    sc   = max(amax / 127.0, 1e-8)
    q    = np.clip(np.round(x / sc), -127.0, 127.0) * sc
    return q, np.float32(sc)


# ═══════════════════════════════════════════════════════════════════════════════
# FIXED-POINT  (port of to_fixed_point)

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
    intervals = np.linspace(-10.0, 0.0, 13)   # 12 intervals → 13 boundaries
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
    Exact port of PLASoftmax.forward():
      max-subtract → Q5.26 → pla_exp → / sum → Quantize.apply
    Returns (float64_on_int8_grid, scale).
    """
    mx      = scores.max(axis=-1, keepdims=True)
    shifted = to_fixed(scores - mx, 32, 26)      # Q5.26
    exps    = _pla_exp(shifted)
    probs   = exps / (exps.sum(axis=-1, keepdims=True) + 1e-9)
    return fake_quant(probs)                      # Quantize.apply inside PLASoftmax


# ═══════════════════════════════════════════════════════════════════════════════
# GCU — GELU COMPUTE UNIT

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
    """Port of ExponentialUnit.forward()."""
    x = x.astype(np.float64)
    if use_log2e:
        x = x * (1.0 + 0.5 - 0.0625)    # log2e ≈ 1.4375
    x_int  = np.floor(x).astype(np.int64)
    x_frac = np.clip(x - x_int.astype(np.float64), 0.0, 0.999999)
    seg    = np.clip(np.floor(x_frac * 8).astype(np.int64), 0, 7)
    K = _EXP_COEFFS[seg, 0];  B = _EXP_COEFFS[seg, 1]
    return (K * x_frac + B) * (2.0 ** np.clip(x_int, -15, 15).astype(np.float64))

def _leading_one(x: np.ndarray):
    """Port of DivisionUnit.leading_one_detector()."""
    abs_x  = np.maximum(np.abs(x.astype(np.float64)), 1e-8)
    log2_x = np.log2(abs_x)
    w      = np.floor(log2_x).astype(np.int64)
    m      = log2_x - w.astype(np.float64) + 1.0
    return w, m

def _div_unit(num: np.ndarray, den: np.ndarray, add_one: bool = False) -> np.ndarray:
    """Port of DivisionUnit.forward()."""
    if add_one:
        den = 1.0 + den
    w1, m1 = _leading_one(num)
    w2, m2 = _leading_one(den)
    exp    = (m1 + w1.astype(np.float64)) - (m2 + w2.astype(np.float64))
    result = _exp_unit(exp, use_log2e=False)
    return result * (np.sign(num) * np.sign(den))

def gcu(x: np.ndarray) -> np.ndarray:
    """
    Port of GCU.forward():
      PolynomialUnit: h_x = 0.8*x + 0.0625*x^3
                      s_x = -10.3125 * h_x
      exp_term = ExponentialUnit(-s_x, log2e=False)
      return DivisionUnit(x, exp_term, add_one=True)  →  x / (1 + exp_term)
    """
    x        = x.astype(np.float64)
    h_x      = 0.8 * x + 0.0625 * (x ** 3)        # PolynomialUnit
    s_x      = -10.3125 * h_x                       # s_x_coeff = -10.0 - 0.25 - 0.0625
    exp_term = _exp_unit(-s_x, use_log2e=False)
    return _div_unit(x, exp_term, add_one=True)


# ═══════════════════════════════════════════════════════════════════════════════
# APPROXIMATE LAYER NORM  (exact port of ApproximateLayerNorm)

def approx_layernorm(x: np.ndarray, gamma: np.ndarray, beta: np.ndarray,
                     eps: float = 1e-12, nr: int = 8) -> np.ndarray:
    """
    Port of ApproximateLayerNorm.forward():
      var → Q5.26 → Newton-Raphson sqrt → normalise → affine
    Returns float64 — caller does NOT apply fake_quant (training does not either).
    """
    mean   = x.mean(axis=-1, keepdims=True)
    var    = x.var(axis=-1,  keepdims=True)     # unbiased=False  (PyTorch default)
    var_fx = to_fixed(var, 32, 26)              # Q5.26

    s = np.where(var_fx > 1.0, var_fx * 0.5, np.ones_like(var_fx))
    for _ in range(nr):
        s = 0.5 * (s + var_fx / (s + 1e-9))

    x_norm = (x - mean) / (s + eps)
    return gamma.astype(np.float64) * x_norm + beta.astype(np.float64)


# ═══════════════════════════════════════════════════════════════════════════════
# QUANTIZED LINEAR  (exact port of QuantizedLinear.forward)
def quantized_linear(x: np.ndarray, W_f: np.ndarray,
                     b_f: np.ndarray = None) -> np.ndarray:
    """
    Port of QuantizedLinear.forward():
      q_input  = Quantize.apply(input)
      q_weight = Quantize.apply(weight)          ← weight already on INT8 grid
      return q_input @ q_weight.T + bias         ← bias is plain float
    W_f is already on the INT8 grid (loaded as INT8, dequanted once at load time),
    so fake_quant(W_f) ≈ W_f — we apply it anyway for exactness.
    b_f is the plain float bias (INT32 * bias_scale at load time).
    """
    x_fq,  _ = fake_quant(x)
    W_fq,  _ = fake_quant(W_f)     # weight is already on INT8 grid; this is a no-op
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

        # ── Q/K/V: QuantizedLinear (fake_quant input + fake_quant weight + float bias)
        q_full = self._split(quantized_linear(x, self.Wq, self.bq), B, S)
        k_full = self._split(quantized_linear(x, self.Wk, self.bk), B, S)
        v_full = self._split(quantized_linear(x, self.Wv, self.bv), B, S)

        # ── QKt tiled: each tile is Quantize.apply'd before matmul
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

        # ── PLASoftmax (includes Quantize.apply inside)
        attn_probs, _ = pla_softmax(scores)

        # ── V: Quantize.apply once on entire v_full, then tiled matmul
        v_fq, _ = fake_quant(v_full)
        D       = self.head_dim
        ctx     = np.zeros((B, H, S, D), np.float64)
        for i in range(0, S, T):
            ctx[:, :, i:i+T, :] = attn_probs[:, :, i:i+T, :] @ v_fq

        # merge heads
        ctx = ctx.transpose(0, 2, 1, 3).reshape(B, S, self.hidden)

        # ── Wo: QuantizedLinear (no extra Quantize.apply in training)
        return quantized_linear(ctx, self.Wo, self.bo)


# ═══════════════════════════════════════════════════════════════════════════════
# FEED-FORWARD  (exact port of FeedForward.forward)

class FFNBlock:
    TILE = 32

    def __init__(self):
        self.W1 = self.W2 = None
        self.b1 = self.b2 = None

    def forward(self, x: np.ndarray):
        B, S, _  = x.shape
        T        = self.TILE

        # ── ffn_1 tiled → GCU → Quantize.apply
        out1 = np.zeros((B, S, self.W1.shape[0]), np.float64)
        for i in range(0, S, T):
            out1[:, i:i+T, :] = quantized_linear(x[:, i:i+T, :], self.W1, self.b1)
        x_mid, _ = fake_quant(gcu(out1))

        # ── ffn_2 tiled  (no Quantize.apply on ffn output in training)
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

        # Residual 1: Quantize.apply(x) + Quantize.apply(attn) → LayerNorm
        # Training: norm_input_1 = Quantize.apply(x) + Quantize.apply(attn_output)
        #           x = attn_layer_norm(norm_input_1)   ← raw LN output, NO extra quant
        x_fq,    _ = fake_quant(x)
        attn_fq, _ = fake_quant(attn_out)
        x = approx_layernorm(x_fq + attn_fq, self.ln1_gamma, self.ln1_beta, eps=1e-12)

        ffn_out = self.ffn.forward(x)

        # Residual 2: same pattern
        x_fq2,   _ = fake_quant(x)
        ffn_fq,  _ = fake_quant(ffn_out)
        x = approx_layernorm(x_fq2 + ffn_fq, self.ln2_gamma, self.ln2_beta, eps=1e-12)

        return x   # raw LayerNorm output — NO Quantize.apply here


# ═══════════════════════════════════════════════════════════════════════════════
# FULL BERT MODEL

class HardwareBERT:
    def __init__(self, num_layers=12, hidden=768, heads=12):
        self.hidden  = hidden
        self.layers  = [EncoderLayer(hidden, heads) for _ in range(num_layers)]
        self.emb_ln_gamma = self.emb_ln_beta = None
        self.word_emb = self.pos_emb = self.tok_type_emb = None
        # pooler: nn.Linear (plain float, not QuantizedLinear)
        self.pooler_W = self.pooler_b = None
        # classifier: QuantizedLinear
        self.clf_W    = self.clf_b    = None

    # ─────────────────────────────────────────────────────────────────────────
    def load_weights(self, npz_path: str):
        print(f"Loading: {npz_path}")
        W = np.load(npz_path, allow_pickle=True)

        def get(k, default=None):
            return W[k] if k in W.files else default

        def w_scale(k):
            v = get(f'{k}.scale');       return float(v) if v is not None else 1.0

        def b_scale(k):
            v = get(f'{k}.scale');       return float(v) if v is not None else 1.0

        def load_weight_f(k):
            """Load INT8 weight → dequant to float (INT8 grid)."""
            raw = get(k)
            if raw is None: return None
            sc = w_scale(k)
            return (raw.astype(np.float32) * sc if raw.dtype == np.int8
                    else raw.astype(np.float32))

        def load_bias_f(k):
            """Load INT32 bias → dequant to float using bias_scale."""
            raw = get(k)
            if raw is None: return None
            sc = b_scale(k)
            return (raw.astype(np.float64) * sc if raw.dtype == np.int32
                    else raw.astype(np.float64))

        def load_emb_f(k):
            """Embeddings are INT8 with a weight scale."""
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

        # Pooler: nn.Linear (plain float — weight stored as INT8, bias as INT32)
        self.pooler_W = load_weight_f('bert.pooler.weight')
        self.pooler_b = load_bias_f('bert.pooler.bias')

        # Classifier: QuantizedLinear
        self.clf_W = load_weight_f('classifier.weight')
        self.clf_b = load_bias_f('classifier.bias')

        print(f"  Loaded {len(W.files)} tensors.")

    # ─────────────────────────────────────────────────────────────────────────
    def forward(self, input_ids, token_type_ids, attention_mask):
        B, S = input_ids.shape
        pos  = np.arange(S)[np.newaxis, :]

        # BertEmbeddings: lookup + add + ApproximateLayerNorm + dropout(noop at inference)
        emb = (self.word_emb[input_ids].astype(np.float64)
             + self.pos_emb[pos].astype(np.float64)
             + self.tok_type_emb[token_type_ids].astype(np.float64))
        x   = approx_layernorm(emb, self.emb_ln_gamma, self.emb_ln_beta, eps=1e-12)

        # Attention mask (matches BertModel.forward)
        mask = (1.0 - attention_mask[:, np.newaxis, np.newaxis, :]) * -10000.0

        # TransformerEncoder
        for layer in self.layers:
            x = layer.forward(x, mask)

        # Pooler: encoder_output[:,0] → nn.Linear (plain float) → Tanh
        cls_f  = x[:, 0, :].astype(np.float64)
        pooled = np.tanh(
            cls_f @ self.pooler_W.T.astype(np.float64) + self.pooler_b.astype(np.float64))

        # Classifier: QuantizedLinear (dropout is noop at inference)
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

    print("\n" + "═"*60)
    print("  HARDWARE-FAITHFUL INT8 BERT  —  SST-2 VALIDATION")
    print("  PLA Softmax | GCU | Newton-Raphson LayerNorm")
    print("═"*60)

    model = HardwareBERT()
    model.load_weights(npz_path)

    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    dataset   = load_dataset("glue", "sst2", split="validation")
    samples   = dataset.select(range(min(num_samples, len(dataset))))
    correct = total = 0

    print(f"\n  Evaluating {len(samples)} samples…\n")
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
    print(f"  Expected  : ~80–88%")
    print("═"*60)
    return acc

if __name__ == "__main__":
    w = r"/kaggle/input/datasets/khalednabil676/sst-2-weight-file/weights_with_scales.npz"
    estimate_accuracy(w)