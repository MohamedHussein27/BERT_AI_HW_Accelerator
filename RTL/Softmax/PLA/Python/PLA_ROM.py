# gen_pla_files_safe2.py
import numpy as np
import math
import sys

# Try to set stdout to UTF-8 if possible (defensive; harmless if not available)
try:
    # Python 3.7+ provides reconfigure; this avoids Windows cp1252 issues
    sys.stdout.reconfigure(encoding='utf-8')
except Exception:
    # ignore if not available
    pass

# config
Q = 26
XMIN = -16.0
XMAX = 16.0
h = 1.0
NSEG = int((XMAX - XMIN) / h)
NUM_TEST = 1000
SEED = 1234

np.random.seed(SEED)

def to_q26_32(x):
    val = int(np.round(x * (1 << Q)))
    # wrap to 32-bit two's complement
    return val & 0xFFFFFFFF

def hex32(v):
    return "{:08x}".format(v & 0xFFFFFFFF)

# compute float PLA coefficients over full domain (we'll write full ROM)
xs = [XMIN + i*h for i in range(NSEG)]
w = []
b = []
for x0 in xs:
    x1 = x0 + h
    m = (math.exp(x1) - math.exp(x0)) / (x1 - x0)
    c = math.exp(x0) - m * x0
    w.append(m)
    b.append(c)

# Convert all coefficients to Q5.26 32-bit (they may overflow for large segments,
# but we still write full ROM; tests will avoid indices causing overflow.)
w_q_full = [to_q26_32(v) for v in w]
b_q_full = [to_q26_32(v) for v in b]

with open('pla_w.hex','w') as fw:
    for v in w_q_full:
        fw.write(hex32(v) + "\n")
with open('pla_b.hex','w') as fb:
    for v in b_q_full:
        fb.write(hex32(v) + "\n")

print("W/B ROMs written: NSEG =", NSEG)

# ---------- determine safe test x-range for 32-bit Q5.26 ----------
MAX_INT32 = (1 << 31) - 1
max_val = MAX_INT32 / float(1 << Q)  # ≈ 32.0

# Condition A: outputs must fit: exp(x) <= max_val  -> x <= ln(max_val)
x_max_out = math.log(max_val)

# Condition B: slopes must fit: slope ≈ e^x * (e-1) <= max_val  -> x <= ln(max_val/(e-1))
x_max_slope = math.log(max_val / (math.e - 1.0))

# choose safe XMAX as the smaller of the two, minus a small guard
XMAX_SAFE = min(x_max_out, x_max_slope) - 0.1
XMIN_SAFE = -XMAX_SAFE

print("max_val (Q5.26 32-bit) ≈", max_val)
print("x_max_out  =", x_max_out)
print("x_max_slope=", x_max_slope)
print("Using SAFE range: [{:.6f}, {:.6f}]".format(XMIN_SAFE, XMAX_SAFE))

# generate test vectors within safe domain
xs_test = np.random.uniform(low=XMIN_SAFE, high=XMAX_SAFE, size=NUM_TEST)
# add some edge cases
xs_test[:6] = np.array([XMIN_SAFE, XMIN_SAFE+1e-6, -1.0, 0.0, 1.0, XMAX_SAFE-1e-6])

inputs_q = [to_q26_32(x) for x in xs_test]
expected_q = [to_q26_32(math.exp(x)) for x in xs_test]

with open('inputs.hex','w') as fi:
    for v in inputs_q:
        fi.write(hex32(v) + "\n")
with open('expected.hex','w') as fe:
    for v in expected_q:
        fe.write(hex32(v) + "\n")

print("inputs/expected written; NUM_TEST =", NUM_TEST)
