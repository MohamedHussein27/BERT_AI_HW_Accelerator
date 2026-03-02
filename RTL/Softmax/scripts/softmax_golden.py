#!/usr/bin/env python3
"""
softmax_golden.py — Python Golden Model for BERT Softmax Hardware Validation.

Generates test vectors for the Softmax hardware module and computes
expected outputs using both floating-point and hardware-emulated fixed-point.

Outputs:
  - input_vectors.hex    : Input test vectors in Q5.26 signed (32-bit)
  - expected_outputs.hex : Expected softmax outputs in Q1.15 unsigned (16-bit)
  - exp_test_inputs.hex  : Test inputs for PLA exp unit test
  - exp_test_expected.hex: Expected exp outputs for unit test
  - recip_test_inputs.hex: Test inputs for reciprocal unit test
  - recip_test_expected.hex: Expected reciprocal outputs for unit test

Also prints error analysis: MSE, max abs error vs PLA segment count.
"""

import numpy as np
import math
import os
import sys

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
# Fixed-point parameters (must match softmax_pkg.sv)
DATA_W      = 32
FRAC_IN     = 26      # Q5.26 input
EXP_W       = 16
FRAC_EXP    = 15      # Q1.15 exp output
ACC_W       = 32
FRAC_ACC    = 24      # Q8.24 accumulator
NORM_W      = 16
FRAC_NORM   = 15      # Q1.15 normalized output

# PLA parameters
PLA_NSEG    = 32
PLA_XMIN    = -16.0
PLA_XMAX    = 0.0
PLA_H       = (PLA_XMAX - PLA_XMIN) / PLA_NSEG

# Test parameters
VEC_LEN     = 512      # BERT full sequence length
NUM_VECTORS = 4        # Number of random test vectors
SEED        = 42

OUTPUT_DIR = os.path.dirname(os.path.abspath(__file__))

np.random.seed(SEED)

# ---------------------------------------------------------------------------
# Fixed-point conversion helpers
# ---------------------------------------------------------------------------
def float_to_q526_signed(val):
    """Float -> Q5.26 signed 32-bit (two's complement)."""
    raw = int(round(val * (1 << FRAC_IN)))
    return raw & 0xFFFFFFFF

def q526_to_float(val_u32):
    """Q5.26 signed 32-bit -> float."""
    if val_u32 & 0x80000000:
        val_s = val_u32 - (1 << DATA_W)
    else:
        val_s = val_u32
    return val_s / float(1 << FRAC_IN)

def float_to_uq115(val):
    """Float -> Q1.15 unsigned 16-bit."""
    if val < 0: val = 0.0
    raw = int(round(val * (1 << FRAC_EXP)))
    return min(raw, (1 << EXP_W) - 1) & 0xFFFF

def uq115_to_float(val_u16):
    """Q1.15 unsigned 16-bit -> float."""
    return val_u16 / float(1 << FRAC_EXP)

def float_to_uq824(val):
    """Float -> Q8.24 unsigned 32-bit."""
    if val < 0: val = 0.0
    raw = int(round(val * (1 << FRAC_ACC)))
    return min(raw, (1 << ACC_W) - 1) & 0xFFFFFFFF

def uq824_to_float(val_u32):
    """Q8.24 unsigned 32-bit -> float."""
    return val_u32 / float(1 << FRAC_ACC)

def hex32(v): return f"{v & 0xFFFFFFFF:08x}"
def hex16(v): return f"{v & 0xFFFF:04x}"

# ---------------------------------------------------------------------------
# PLA Exp Approximation (emulating hardware)
# ---------------------------------------------------------------------------
def pla_exp_hw(x_float, nseg=PLA_NSEG, xmin=PLA_XMIN, xmax=PLA_XMAX):
    """Emulate hardware PLA exp computation using local offset formulation.
    
    exp(x) ≈ slope[idx] * (x - x0) + exp(x0)
    where x0 = xmin + idx * h (segment start)
    """
    h = (xmax - xmin) / nseg

    # Clamp
    x = max(xmin, min(xmax, x_float))

    # Segment index
    idx = int((x - xmin) / h)
    idx = min(idx, nseg - 1)

    # Segment start
    x0 = xmin + idx * h
    x1 = x0 + h

    # Compute slope and intercept
    exp_x0 = math.exp(x0)
    exp_x1 = math.exp(x1)
    w = (exp_x1 - exp_x0) / (x1 - x0)  # slope
    b = exp_x0                            # intercept = exp(x0)

    # Quantize coefficients to Q1.15
    w_q = float_to_uq115(w)
    b_q = float_to_uq115(b)

    # Compute local offset using bit-mask (matching hardware)
    # Hardware computes delta = x_q - XMIN_Q, then x_local = delta[H_SHIFT-1:0]
    H_SHIFT = 25  # For h = 0.5
    x_q = float_to_q526_signed(x)
    x_s = x_q if x_q < (1 << 31) else x_q - (1 << 32)
    xmin_q = int(xmin * (1 << FRAC_IN))
    delta = x_s - xmin_q  # Always positive
    mask = (1 << H_SHIFT) - 1
    x_local_q = delta & mask  # Lower H_SHIFT bits = local offset in Q5.26

    # Product: w_q (16b unsigned) * x_local_q (32b unsigned) = 48b unsigned
    product = w_q * x_local_q
    # Shift right by FRAC_IN (26) to get Q1.15
    scaled = product >> FRAC_IN

    result = scaled + b_q

    # Clamp
    if result < 0: result = 0
    if result > 0xFFFF: result = 0xFFFF

    return result & 0xFFFF


# ---------------------------------------------------------------------------
# Newton-Raphson Reciprocal Emulation (matches RTL exactly)
# ---------------------------------------------------------------------------
NR_LUT_BITS = 4
NR_LUT_SIZE = 1 << NR_LUT_BITS
NR_ITER = 2

# Load NR LUT from hex file
def load_nr_lut():
    lut = []
    lut_path = os.path.join(OUTPUT_DIR, "nr_init.hex")
    with open(lut_path) as f:
        for line in f:
            line = line.strip()
            if line:
                lut.append(int(line, 16))
    return lut

def nr_reciprocal_hw(a_q):
    """Emulate RTL Newton-Raphson reciprocal with normalization.
    
    Input/output in Q8.24 unsigned 32-bit.
    """
    MASK32 = 0xFFFFFFFF
    MASK64 = 0xFFFFFFFFFFFFFFFF
    TWO_Q = (2 << FRAC_ACC) & MASK32  # 2.0 in Q8.24
    
    if a_q == 0:
        return MASK32
    
    # Step 1: Find MSB of integer part
    int_part = (a_q >> FRAC_ACC) & 0xFF
    msb_pos = 0
    for bit in range(7, -1, -1):
        if int_part & (1 << bit):
            msb_pos = bit
            break
    
    # Step 2: Normalize — shift right by msb_pos
    if int_part == 0:
        a_norm = a_q
        shift_amt = 0
    else:
        a_norm = (a_q >> msb_pos) & MASK32
        shift_amt = msb_pos
    
    # Step 3: LUT lookup using top 4 frac bits of normalized value
    lut = load_nr_lut()
    lut_idx = (a_norm >> (FRAC_ACC - NR_LUT_BITS)) & ((1 << NR_LUT_BITS) - 1)
    y = lut[lut_idx]
    
    # Step 4: NR iterations (matches RTL 3-stage FSM exactly)
    for _ in range(NR_ITER):
        # MUL_AY: a_norm * y in 64 bits
        ay_64 = (a_norm * y) & MASK64
        
        # SUB: 2 - (a*y >> Q)
        ay_scaled = (ay_64 >> FRAC_ACC) & MASK32
        sub = (TWO_Q - ay_scaled) & MASK32
        
        # UPDATE: y = y * sub >> Q
        ysub_64 = (y * sub) & MASK64
        y = (ysub_64 >> FRAC_ACC) & MASK32
    
    # Step 5: De-normalize — shift right by shift_amt
    result = (y >> shift_amt) & MASK32
    return result


# ---------------------------------------------------------------------------
# Numerically Stable Softmax (Float Reference)
# ---------------------------------------------------------------------------
def softmax_float(vec):
    """Standard numerically stable softmax in floating point."""
    m = max(vec)
    shifted = [x - m for x in vec]
    exps = [math.exp(x) for x in shifted]
    s = sum(exps)
    return [e / s for e in exps]

# ---------------------------------------------------------------------------
# Hardware-Emulated Softmax
# ---------------------------------------------------------------------------
def softmax_hw(vec):
    """Emulate the full hardware softmax pipeline in fixed-point matching RTL."""
    n = len(vec)
    MASK32 = 0xFFFFFFFF
    
    # Step 0: Convert all inputs to Q5.26 signed (same as RTL input)
    inputs_q = [float_to_q526_signed(x) for x in vec]
    
    # Step 1: Find max in Q5.26 (signed comparison)
    def signed32(v):
        return v - (1 << 32) if v & 0x80000000 else v
    
    max_q = inputs_q[0]
    for i in range(1, n):
        if signed32(inputs_q[i]) > signed32(max_q):
            max_q = inputs_q[i]
    
    # Step 2: Subtract max in Q5.26 (signed subtraction)
    shifted_q = []
    for x_q in inputs_q:
        diff = (signed32(x_q) - signed32(max_q)) & MASK32
        shifted_q.append(diff)
    
    # Step 3: PLA exp — each shifted value goes through PLA
    # Convert Q5.26 signed to float for PLA (which is already emulated in fixed-point)
    exp_vals_q = [pla_exp_hw(q526_to_float(sq)) for sq in shifted_q]

    # Step 4: Accumulate (align to Q8.24 and sum)
    ALIGN_SHIFT = FRAC_ACC - FRAC_EXP  # 9
    acc = 0
    for e in exp_vals_q:
        acc += e << ALIGN_SHIFT
    acc = min(acc, (1 << ACC_W) - 1)  # Saturate

    # Step 5: NR Reciprocal (matches RTL exactly)
    recip_q = nr_reciprocal_hw(acc)

    # Step 6: Normalize
    outputs = []
    for e in exp_vals_q:
        product = e * recip_q
        result = (product >> FRAC_ACC) & 0xFFFF
        outputs.append(result)

    return outputs, exp_vals_q, acc

# ---------------------------------------------------------------------------
# Generate test vectors
# ---------------------------------------------------------------------------
print("=" * 70)
print("BERT Softmax Golden Model — Test Vector Generation")
print("=" * 70)
print()

# Generate random input vectors in typical BERT attention score range
# After QK^T / sqrt(d_k), scores are typically in [-5, 5]
test_vectors = []
for v in range(NUM_VECTORS):
    if v == 0:
        # Edge case: all same values
        vec = [0.0] * VEC_LEN
    elif v == 1:
        # Edge case: one hot (one large, rest small)
        vec = [-10.0] * VEC_LEN
        vec[0] = 0.0
    elif v == 2:
        # Edge case: linearly varying
        vec = [i * 0.02 - 5.0 for i in range(VEC_LEN)]
    else:
        # Random in typical BERT range
        vec = np.random.uniform(-4.0, 4.0, VEC_LEN).tolist()
    test_vectors.append(vec)

# ---------------------------------------------------------------------------
# Compute golden outputs and write hex files
# ---------------------------------------------------------------------------
all_hw_errors = []

input_hex_path = os.path.join(OUTPUT_DIR, "input_vectors.hex")
output_hex_path = os.path.join(OUTPUT_DIR, "expected_outputs.hex")

with open(input_hex_path, 'w') as f_in, open(output_hex_path, 'w') as f_out:
    for v_idx, vec in enumerate(test_vectors):
        # Float reference
        sm_float = softmax_float(vec)

        # HW emulation
        sm_hw, exp_vals, acc = softmax_hw(vec)

        # Write input vector
        for x in vec:
            f_in.write(hex32(float_to_q526_signed(x)) + "\n")

        # Write expected output
        for y in sm_hw:
            f_out.write(hex16(y) + "\n")

        # Compute errors
        for i in range(len(vec)):
            hw_val = uq115_to_float(sm_hw[i])
            err = abs(hw_val - sm_float[i])
            all_hw_errors.append(err)

        # Print summary for this vector
        sm_hw_float = [uq115_to_float(y) for y in sm_hw]
        mse_vec = np.mean([(a - b)**2 for a, b in zip(sm_float, sm_hw_float)])
        max_err_vec = max(abs(a - b) for a, b in zip(sm_float, sm_hw_float))
        sum_hw = sum(sm_hw_float)
        print(f"Vector {v_idx}: MSE={mse_vec:.2e}, MaxErr={max_err_vec:.4f}, Sum={sum_hw:.6f}")

# ---------------------------------------------------------------------------
# Generate unit test vectors for PLA exp
# ---------------------------------------------------------------------------
print()
print("Generating PLA exp unit test vectors...")
NUM_EXP_TESTS = 200
exp_test_inputs = np.random.uniform(-16.0, 0.0, NUM_EXP_TESTS).tolist()
# Add edge cases
exp_test_inputs[:5] = [-16.0, -15.0, -8.0, -1.0, 0.0]

exp_in_path = os.path.join(OUTPUT_DIR, "exp_test_inputs.hex")
exp_exp_path = os.path.join(OUTPUT_DIR, "exp_test_expected.hex")

with open(exp_in_path, 'w') as f_in, open(exp_exp_path, 'w') as f_exp:
    for x in exp_test_inputs:
        f_in.write(hex32(float_to_q526_signed(x)) + "\n")
        result = pla_exp_hw(x)
        f_exp.write(hex16(result) + "\n")

print(f"  Written {NUM_EXP_TESTS} test vectors")

# ---------------------------------------------------------------------------
# Generate unit test vectors for reciprocal
# ---------------------------------------------------------------------------
print()
print("Generating reciprocal unit test vectors...")
NUM_RECIP_TESTS = 50
recip_test_vals = np.random.uniform(1.0, 128.0, NUM_RECIP_TESTS).tolist()
recip_test_vals[:4] = [1.0, 2.0, 32.0, 64.0]

recip_in_path = os.path.join(OUTPUT_DIR, "recip_test_inputs.hex")
recip_exp_path = os.path.join(OUTPUT_DIR, "recip_test_expected.hex")

with open(recip_in_path, 'w') as f_in, open(recip_exp_path, 'w') as f_exp:
    for a in recip_test_vals:
        a_q = float_to_uq824(a)
        r_q = float_to_uq824(1.0 / a)
        f_in.write(hex32(a_q) + "\n")
        f_exp.write(hex32(r_q) + "\n")

print(f"  Written {NUM_RECIP_TESTS} test vectors")

# ---------------------------------------------------------------------------
# Error analysis: MSE vs segment count
# ---------------------------------------------------------------------------
print()
print("=" * 70)
print("Error Analysis: PLA Accuracy vs Segment Count")
print("=" * 70)
print()
print(f"{'Segments':>8s} | {'h':>8s} | {'MSE':>12s} | {'Max Abs Err':>12s} | {'Max Rel Err':>12s}")
print("-" * 60)

test_x = np.linspace(-10.0, 0.0, 1000)  # Focus on active range

for nseg in [8, 16, 32, 64, 128]:
    abs_errors = []
    rel_errors = []
    for x in test_x:
        hw_result = uq115_to_float(pla_exp_hw(x, nseg=nseg))
        true_result = math.exp(x)
        ae = abs(hw_result - true_result)
        re = ae / max(true_result, 1e-12)
        abs_errors.append(ae)
        rel_errors.append(re)

    mse = np.mean([e**2 for e in abs_errors])
    max_ae = max(abs_errors)
    max_re = max(rel_errors)
    h_seg = 16.0 / nseg
    print(f"{nseg:8d} | {h_seg:8.4f} | {mse:12.2e} | {max_ae:12.6f} | {max_re:12.6f}")

# ---------------------------------------------------------------------------
# Overall statistics
# ---------------------------------------------------------------------------
print()
print("=" * 70)
print("Overall Softmax Pipeline Error (with 32 segments)")
print("=" * 70)
print(f"  Number of vectors:   {NUM_VECTORS}")
print(f"  Vector length:       {VEC_LEN}")
print(f"  Total elements:      {len(all_hw_errors)}")
print(f"  Mean Absolute Error: {np.mean(all_hw_errors):.6f}")
print(f"  Max Absolute Error:  {max(all_hw_errors):.6f}")
print(f"  MSE:                 {np.mean([e**2 for e in all_hw_errors]):.2e}")
print()

# ---------------------------------------------------------------------------
# Performance estimates
# ---------------------------------------------------------------------------
print("=" * 70)
print("Performance & Resource Estimates")
print("=" * 70)
print(f"""
  Latency per {VEC_LEN}-element vector:
    Phase 1 (Max):         {VEC_LEN} clocks
    Phase 2 (Sub+Exp+Acc): {VEC_LEN} + 5 pipeline clocks = {VEC_LEN + 5}
    Phase 3 (Reciprocal):  ~8 clocks (capture + 2 NR iterations)
    Phase 4 (Normalize):   {VEC_LEN} + 2 pipeline clocks = {VEC_LEN + 2}
    Total:                 ~{VEC_LEN * 3 + 15} clocks

  Throughput at 100 MHz:
    Clocks per vector:     {VEC_LEN * 3 + 15}
    Time per vector:       {(VEC_LEN * 3 + 15) * 10:.0f} ns = {(VEC_LEN * 3 + 15) * 0.01:.2f} µs
    Vectors per second:    {100e6 / (VEC_LEN * 3 + 15):.0f}

  Estimated FPGA Resources (Artix-7):
    DSP48E1:               3-4 (PLA multiply, NR multiplies, normalize)
    LUTs:                  ~2000-2500
    FFs:                   ~800-1200
    BRAM (36Kb):           0 (distributed RAM sufficient)
    Fmax:                  >150 MHz (single-multiply critical path)

  FPGA Optimization Strategy:
    1. Use DSP48E1 for all multiplies (inferred automatically)
    2. ROM fits in distributed LUTRAMs (32x16b = 512 bits)
    3. SRAM fits in distributed RAM (128x32b = 4Kb)
    4. Pipeline registers break critical paths
    5. No BRAM needed — saves routing resources
    6. Can process multiple rows in parallel by instantiating N copies

  Integration into BERT Attention Pipeline:
    1. Connect QK^T output (after scaling by 1/sqrt(d_k)) to bert_softmax input
    2. Each attention head has its own softmax instance
    3. Output feeds into Value multiplication stage
    4. For multi-head attention: instantiate {12} copies (BERT-base has 12 heads)
       or time-multiplex with a shared softmax engine
""")
