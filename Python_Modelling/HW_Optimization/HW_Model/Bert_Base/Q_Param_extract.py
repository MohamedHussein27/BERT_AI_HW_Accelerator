import math
import re
from collections import defaultdict

def calc_m0_s(m):
    """Converts a float M into integer multiplier M0 and shift S."""
    if m <= 0: return 0, 0
    frac, exp = math.frexp(m)
    m0 = round(frac * (2**31))
    s = -exp
    return m0, s

# ---------------------------------------------------------
# Box Mappers (Matching the Flow Diagram)
# ---------------------------------------------------------

def box_qkv_quantize(in_scale, w_scale, out_scale):
    """Q/K/V-quantize 32 int to 8"""
    return calc_m0_s((in_scale * w_scale) / out_scale)

def box_qkt_dequantize(scale_q, scale_k, d_k=64):
    """dequantization 32 int to Q5.26 (Pre-Softmax)"""
    return calc_m0_s((scale_q * scale_k * (1.0 / math.sqrt(d_k))) * (2**26))

def box_softmax_quantize(scale_softmax_out):
    """Quantization Q1.15 to 8 bit (Post-Softmax)"""
    return calc_m0_s((2**-15) / scale_softmax_out)

def box_vgemm_quantize(scale_softmax_out, scale_v_out, scale_ctx_out):
    """Quantization 32 int to 8 (.VGeMM)"""
    return calc_m0_s((scale_softmax_out * scale_v_out) / scale_ctx_out)

def box_wo_quantize(scale_ctx_out, w_scale_wo, scale_out_out):
    """Quantization 32 int to 8 (WoGeMM)"""
    return calc_m0_s((scale_ctx_out * w_scale_wo) / scale_out_out)

def box_ln_quantize(scale_ln_out):
    """Quantization ? to 8 (Post Add&Norm)
    Assuming the Add&Norm outputs Q5.26 from the previous step."""
    return calc_m0_s((2**-26) / scale_ln_out)

def box_ffn1_dequantize(scale_ln_out, w_scale_ffn1):
    """dequantization 32 to Q10.22 (Pre-GELU)"""
    return calc_m0_s((scale_ln_out * w_scale_ffn1) * (2**22))

def box_gelu_quantize(scale_gcu_out):
    """Quantization Q48.16 to 8 bit (Post-GELU)"""
    # UPDATED: Changed from 2**-12 to 2**-16
    return calc_m0_s((2**-16) / scale_gcu_out)

def box_ffn2_quantize(scale_gcu_out, w_scale_ffn2, scale_ffn2_out):
    """Quantization 32 int to 8 (.ffn_2GeMM)"""
    return calc_m0_s((scale_gcu_out * w_scale_ffn2) / scale_ffn2_out)


def main():
    # 1. Read the raw log file
    try:
        with open('raw_scales.log', 'r') as f:
            log_data = f.read()
    except FileNotFoundError:
        print("Error: Save your scales log as 'raw_scales.log'")
        return

    layers = defaultdict(lambda: defaultdict(dict))

    # 2. Parse Weights and Input Scales
    w_pattern = re.compile(r"bert\.encoder\.layers\.(\d+)\.(attention\.[qkv]|attention\.out|ffn\.dense_1|ffn\.dense_2)\.weight:\s*weight_scale=([0-9eE\.\+\-]+),\s*input_scale=([0-9eE\.\+\-]+)")
    for match in w_pattern.finditer(log_data):
        l_id, mod, w_s, in_s = match.groups()
        layers[int(l_id)][mod]['w_scale'] = float(w_s)
        layers[int(l_id)][mod]['in_scale'] = float(in_s)

    # 3. Parse Output (Activation) Scales
    act_pattern = re.compile(r"act_scale\s*bert\.encoder\.layers\.(\d+)\.(attention\.[qkv]|attention\.softmax|attention\.ctx|attention\.out|ffn\.gcu|ffn\.dense_2|attn_layer_norm|ffn_layer_norm)\.output_scale:\s*([0-9eE\.\+\-]+)")
    for match in act_pattern.finditer(log_data):
        l_id, mod, out_s = match.groups()
        layers[int(l_id)][mod]['out_scale'] = float(out_s)

    # 4. Generate the Hardware Config Log
    with open('hardware_config.log', 'w') as f:
        f.write("=== ACCELERATOR QUANTIZATION PIPELINE M0/S CONFIG ===\n\n")

        for l_id in sorted(layers.keys()):
            l = layers[l_id]
            f.write(f"--------------------- LAYER {l_id} ---------------------\n")

            # Q, K, V Projections
            for p in ['attention.q', 'attention.k', 'attention.v']:
                if p in l and 'w_scale' in l[p] and 'out_scale' in l[p]:
                    m0, s = box_qkv_quantize(l[p]['in_scale'], l[p]['w_scale'], l[p]['out_scale'])
                    f.write(f"[{p.upper()[:5]}-Quantize (32->8)]     M0: {m0:<12} | S: {s}\n")

            # QKt GeMM to Q5.26
            if 'attention.q' in l and 'attention.k' in l:
                m0, s = box_qkt_dequantize(l['attention.q']['out_scale'], l['attention.k']['out_scale'])
                f.write(f"[QKt Dequantize (32->Q5.26)] M0: {m0:<12} | S: {s}\n")

            # Softmax to 8-bit
            if 'attention.softmax' in l:
                m0, s = box_softmax_quantize(l['attention.softmax']['out_scale'])
                f.write(f"[Softmax Quant (Q1.15->8)]   M0: {m0:<12} | S: {s}\n")

            # .VGeMM to 8-bit
            if 'attention.softmax' in l and 'attention.v' in l and 'attention.ctx' in l:
                m0, s = box_vgemm_quantize(l['attention.softmax']['out_scale'], l['attention.v']['out_scale'], l['attention.ctx']['out_scale'])
                f.write(f"[.VGeMM Quant (32->8)]       M0: {m0:<12} | S: {s}\n")

            # WoGeMM to 8-bit
            if 'attention.out' in l and 'attention.ctx' in l:
                m0, s = box_wo_quantize(l['attention.ctx']['out_scale'], l['attention.out']['w_scale'], l['attention.out']['out_scale'])
                f.write(f"[Wo Quantize (32->8)]        M0: {m0:<12} | S: {s}\n")

            # LayerNorm to 8-bit
            if 'attn_layer_norm' in l:
                m0, s = box_ln_quantize(l['attn_layer_norm']['out_scale'])
                f.write(f"[Add&Norm Quant (?->8)]      M0: {m0:<12} | S: {s}\n")

            # FFN1 to Q10.22 (Pre-GELU)
            if 'ffn.dense_1' in l and 'attn_layer_norm' in l:
                m0, s = box_ffn1_dequantize(l['attn_layer_norm']['out_scale'], l['ffn.dense_1']['w_scale'])
                f.write(f"[FFN1 Dequant (32->Q10.22)]  M0: {m0:<12} | S: {s}\n")

            # GELU to 8-bit (UPDATED to Q48.16)
            if 'ffn.gcu' in l:
                m0, s = box_gelu_quantize(l['ffn.gcu']['out_scale'])
                f.write(f"[GELU Quant (Q48.16->8)]     M0: {m0:<12} | S: {s}\n")

            # FFN2 to 8-bit
            if 'ffn.dense_2' in l and 'ffn.gcu' in l:
                m0, s = box_ffn2_quantize(l['ffn.gcu']['out_scale'], l['ffn.dense_2']['w_scale'], l['ffn.dense_2']['out_scale'])
                f.write(f"[FFN2 Quant (32->8)]         M0: {m0:<12} | S: {s}\n")

            f.write("\n")
            
    print("Success! Generated 'hardware_config.log'")

if __name__ == "__main__":
    main()