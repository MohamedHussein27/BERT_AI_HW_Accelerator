# test_bert_int8.py
"""
Test the trained BERT model using true INT8 quantized data without fake quantization.
This script is fully corrected to be compatible with the output of the final BERT_HW.py,
including the GCU activation and all necessary fixes for runtime and key errors.
"""
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from transformers import BertTokenizer
from datasets import load_dataset
import evaluate
from tqdm.auto import tqdm
import numpy as np

# ==============================================================================
# INT8 Quantized Layers (Corrected for Hardware Simulation)
# ==============================================================================

class INT8QuantizedLinear(nn.Module):
    def __init__(self, weight_int8, weight_scale, bias, bias_scale):
        super().__init__()
        self.register_buffer('weight_int8', torch.from_numpy(weight_int8))
        self.register_buffer('weight_scale', torch.from_numpy(weight_scale).float())
        if bias is not None:
            self.register_buffer('bias', torch.from_numpy(bias).float())
            self.register_buffer('bias_scale', torch.from_numpy(bias_scale).float())
        else:
            self.register_buffer('bias', None)
            self.register_buffer('bias_scale', None)

    def forward(self, x):
        qmax = 127.0
        abs_max = x.abs().max(dim=-1, keepdim=True)[0]
        input_scale = abs_max / qmax
        input_scale[input_scale == 0] = 1.0
        x_int8 = torch.clamp(torch.round(x / input_scale), -128.0, 127.0).to(torch.int8)
        output_int32 = F.linear(x_int8.float(), self.weight_int8.float())
        output_float = output_int32.float() * self.weight_scale * input_scale
        if self.bias is not None:
             output_float += self.bias
        return output_float

class INT8PLASoftmax(nn.Module):
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
        exps = self.pla_exp(shifted_scores)
        return exps / (exps.sum(dim=-1, keepdim=True) + 1e-9)

class INT8ApproximateLayerNorm(nn.Module):
    def __init__(self, weight, bias, eps=1e-12, nr_iterations=8):
        super().__init__()
        self.eps = eps
        self.register_buffer('weight', torch.from_numpy(weight).float())
        self.register_buffer('bias', torch.from_numpy(bias).float())
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
        x = (x - mean) / (std_approx + self.eps)
        return self.weight * x + self.bias

# ==============================================================================
# --- GCU (GELU COMPUTE UNIT) AND ITS COMPONENTS (for consistency with training script) ---
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
        s_x = self.polynomial_unit(x)
        exp_term = self.eu(-s_x, use_log2e_scaling=False)
        return self.du(x, exp_term, add_one_to_denominator=True)


# ==============================================================================
# INT8 BERT Model (Full Corrected Architecture)
# ==============================================================================

class INT8BertEmbeddings(nn.Module):
    def __init__(self, params):
        super().__init__()
        self.register_buffer('word_embeddings_weight', torch.from_numpy(params['word_embeddings.weight']))
        self.register_buffer('word_embeddings_scale', torch.from_numpy(params['word_embeddings.weight.scale']).float())
        self.register_buffer('position_embeddings_weight', torch.from_numpy(params['position_embeddings.weight']))
        self.register_buffer('position_embeddings_scale', torch.from_numpy(params['position_embeddings.weight.scale']).float())
        self.register_buffer('token_type_embeddings_weight', torch.from_numpy(params['token_type_embeddings.weight']))
        self.register_buffer('token_type_embeddings_scale', torch.from_numpy(params['token_type_embeddings.weight.scale']).float())
        self.LayerNorm = INT8ApproximateLayerNorm(params['LayerNorm.weight'], params['LayerNorm.bias'])
        self.dropout = nn.Dropout(0.1)

    def forward(self, input_ids, token_type_ids=None):
        seq_len = input_ids.size(1)
        if token_type_ids is None: token_type_ids = torch.zeros_like(input_ids)
        pos_ids = torch.arange(seq_len, dtype=torch.long, device=input_ids.device).unsqueeze(0).expand_as(input_ids)

        word_emb_int = F.embedding(input_ids, self.word_embeddings_weight)
        pos_emb_int = F.embedding(pos_ids, self.position_embeddings_weight)
        token_type_emb_int = F.embedding(token_type_ids, self.token_type_embeddings_weight)

        word_embeddings = word_emb_int.float() * self.word_embeddings_scale
        position_embeddings = pos_emb_int.float() * self.position_embeddings_scale
        token_type_embeddings = token_type_emb_int.float() * self.token_type_embeddings_scale

        embeddings = word_embeddings + position_embeddings + token_type_embeddings
        return self.dropout(self.LayerNorm(embeddings))

class INT8MultiHeadSelfAttention(nn.Module):
    def __init__(self, params, hidden_size=768, num_attention_heads=12):
        super().__init__()
        self.num_heads = num_attention_heads
        self.head_dim = hidden_size // num_attention_heads
        self.scale = self.head_dim ** -0.5
        self.q = INT8QuantizedLinear(params['q.weight'], params['q.weight.scale'], params.get('q.bias'), params.get('q.bias.scale'))
        self.k = INT8QuantizedLinear(params['k.weight'], params['k.weight.scale'], params.get('k.bias'), params.get('k.bias.scale'))
        self.v = INT8QuantizedLinear(params['v.weight'], params['v.weight.scale'], params.get('v.bias'), params.get('v.bias.scale'))
        self.out = INT8QuantizedLinear(params['out.weight'], params['out.weight.scale'], params.get('out.bias'), params.get('out.bias.scale'))
        self.softmax = INT8PLASoftmax()
        self.attn_dropout = nn.Dropout(0.1)
        self.proj_dropout = nn.Dropout(0.1)

    def transpose_for_scores(self, x):
        return x.view(x.size()[:-1] + (self.num_heads, self.head_dim)).permute(0, 2, 1, 3)

    def forward(self, hidden_states, attention_mask=None):
        q = self.transpose_for_scores(self.q(hidden_states))
        k = self.transpose_for_scores(self.k(hidden_states))
        v = self.transpose_for_scores(self.v(hidden_states))
        attn_scores = torch.matmul(q, k.transpose(-1, -2)) * self.scale
        if attention_mask is not None: attn_scores = attn_scores + attention_mask
        attn_probs = self.attn_dropout(self.softmax(attn_scores))
        context = torch.matmul(attn_probs, v).permute(0, 2, 1, 3).contiguous().view(hidden_states.size())
        return self.proj_dropout(self.out(context))

class INT8FeedForward(nn.Module):
    def __init__(self, params):
        super().__init__()
        self.dense_1 = INT8QuantizedLinear(params['dense_1.weight'], params['dense_1.weight.scale'], params.get('dense_1.bias'), params.get('dense_1.bias.scale'))
        self.dense_2 = INT8QuantizedLinear(params['dense_2.weight'], params['dense_2.weight.scale'], params.get('dense_2.bias'), params.get('dense_2.bias.scale'))
        self.gcu = GCU()
        self.dropout = nn.Dropout(0.1)

    def forward(self, x):
        x = self.gcu(self.dense_1(x))
        x = self.dense_2(x)
        return self.dropout(x)

class INT8TransformerEncoderLayer(nn.Module):
    def __init__(self, params):
        super().__init__()
        attn_params = {k.replace('attention.', ''): v for k, v in params.items() if k.startswith('attention.')}
        ffn_params = {k.replace('ffn.', ''): v for k, v in params.items() if k.startswith('ffn.')}
        self.attention = INT8MultiHeadSelfAttention(attn_params)
        self.attn_layer_norm = INT8ApproximateLayerNorm(params['attn_layer_norm.weight'], params['attn_layer_norm.bias'])
        self.ffn = INT8FeedForward(ffn_params)
        self.ffn_layer_norm = INT8ApproximateLayerNorm(params['ffn_layer_norm.weight'], params['ffn_layer_norm.bias'])

    def forward(self, x, attention_mask=None):
        attn_output = self.attention(x, attention_mask=attention_mask)
        x = self.attn_layer_norm(x + attn_output)
        ffn_output = self.ffn(x)
        x = self.ffn_layer_norm(x + ffn_output)
        return x

class INT8BertForSequenceClassification(nn.Module):
    def __init__(self, all_params, hidden_size=768, num_layers=12):
        super().__init__()
        embedding_params = {k.replace('bert.embeddings.', ''): v for k, v in all_params.items() if k.startswith('bert.embeddings.')}
        classifier_params = {k.replace('classifier.', ''): v for k, v in all_params.items() if k.startswith('classifier.')}
        pooler_params = {k.replace('bert.pooler.', ''): v for k, v in all_params.items() if k.startswith('bert.pooler.')}

        self.embeddings = INT8BertEmbeddings(embedding_params)
        self.layers = nn.ModuleList([
            INT8TransformerEncoderLayer({k.replace(f'bert.encoder.layers.{i}.', ''): v for k, v in all_params.items() if k.startswith(f'bert.encoder.layers.{i}.')})
            for i in range(num_layers)
        ])
        self.pooler = nn.Linear(hidden_size, hidden_size)
        self.pooler.weight = nn.Parameter(torch.from_numpy(pooler_params['weight']).float(), requires_grad=False)
        self.pooler.bias = nn.Parameter(torch.from_numpy(pooler_params['bias']).float(), requires_grad=False)
        self.pooler_activation = nn.Tanh()
        self.classifier = INT8QuantizedLinear(classifier_params['weight'], classifier_params['weight.scale'], classifier_params.get('bias'), classifier_params.get('bias.scale'))
        self.dropout = nn.Dropout(0.1)

    def forward(self, input_ids, attention_mask=None):
        extended_mask = (1.0 - attention_mask.unsqueeze(1).unsqueeze(2)) * -10000.0 if attention_mask is not None else None
        hidden_states = self.embeddings(input_ids)
        for layer in self.layers:
            hidden_states = layer(hidden_states, attention_mask=extended_mask)
        pooled_output = self.pooler_activation(self.pooler(hidden_states[:, 0]))
        return self.classifier(self.dropout(pooled_output))

# ==============================================================================
# Testing Function
# ==============================================================================
def test_int8_model():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # --- CHANGE: Use an ABSOLUTE path to the weights file ---
    # Replace this with the full path to your project folder.
    weights_file = 'C:/Users/4 you/Documents/Projects_Reports/Digital_Design/visual_studio_work/AI_Accelerator/20_10/bert_base_hw_quantized_weights.npz'
    
    print(f"Loading quantized parameters from {weights_file}...")
    try:
        loaded_params = np.load(weights_file)
        all_params = {key: loaded_params[key] for key in loaded_params}
    except FileNotFoundError:
        print(f"ERROR: Weight file not found at '{weights_file}'.")
        print("Please run the training script first to generate the weights file.")
        return

    print("Creating INT8 model...")
    model = INT8BertForSequenceClassification(all_params).to(device).eval()
    
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    raw_datasets = load_dataset("glue", "sst2")
    
    def tokenize_function(examples):
        return tokenizer(examples["sentence"], padding="max_length", truncation=True, max_length=128)
    
    tokenized_datasets = raw_datasets.map(tokenize_function, batched=True).remove_columns(["sentence", "idx"]).rename_column("label", "labels")
    tokenized_datasets.set_format("torch")
    
    eval_dataloader = DataLoader(tokenized_datasets["validation"], batch_size=16)
    accuracy_metric = evaluate.load("accuracy")
    
    print("Testing INT8 model...")
    progress_bar = tqdm(eval_dataloader, desc="Evaluating")
    with torch.no_grad():
        for batch in progress_bar:
            batch = {k: v.to(device) for k, v in batch.items()}
            logits = model(batch["input_ids"], attention_mask=batch["attention_mask"])
            predictions = torch.argmax(logits, dim=-1)
            accuracy_metric.add_batch(predictions=predictions, references=batch["labels"])
    
    eval_metric = accuracy_metric.compute()
    print(f"\nINT8 Model Validation Accuracy: {eval_metric['accuracy']:.4f}")
    
    print(f"\nMemory Analysis:")
    fp32_model_size_mb = (109_482_240 * 4) / (1024 * 1024)
    print(f"Original FP32 model size: ~{fp32_model_size_mb:.2f}MB")
    
    total_int8_bytes = sum(param.nbytes for key, param in all_params.items() if '.scale' not in key)
    int8_size_mb = total_int8_bytes / (1024 * 1024)
    print(f"Quantized INT8 model size (weights only): ~{int8_size_mb:.2f}MB")
    print(f"Effective compression ratio: ~{fp32_model_size_mb/int8_size_mb:.2f}x")

if __name__ == "__main__":
    test_int8_model()

