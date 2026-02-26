#!/usr/bin/env python3
"""
gen_pla_rom.py — Generate PLA slope/intercept ROM hex files for Softmax exp approximation.

Domain: [-16, 0] (after max subtraction, all exp arguments are <= 0)
Segment width h = 0.5 (default, 32 segments)

For each segment [x_k, x_{k+1}]:
  slope     w_k = (exp(x_{k+1}) - exp(x_k)) / h
  intercept b_k = exp(x_k) - w_k * x_k

Coefficients stored as unsigned Q1.15 (16-bit).

Output files:
  pla_slopes.hex     — 32 entries, 4-digit hex (16-bit)
  pla_intercepts.hex — 32 entries, 4-digit hex (16-bit)
"""

import numpy as np
import math
import os
import sys

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
Q       = 15          # Fractional bits for coefficients (Q1.15)
COEFF_W = 16          # Coefficient width in bits
XMIN    = -16.0       # Domain minimum
XMAX    = 0.0         # Domain maximum
NSEG    = 32          # Number of segments
h       = (XMAX - XMIN) / NSEG  # Segment width = 0.5

OUTPUT_DIR = os.path.dirname(os.path.abspath(__file__))

# ---------------------------------------------------------------------------
# Conversion helpers
# ---------------------------------------------------------------------------
def float_to_uq115(val):
    """Convert non-negative float to unsigned Q1.15 (16-bit)."""
    if val < 0:
        val = 0.0
    raw = int(round(val * (1 << Q)))
    if raw > (1 << COEFF_W) - 1:
        raw = (1 << COEFF_W) - 1  # Saturate
    return raw & ((1 << COEFF_W) - 1)

def hex16(v):
    return f"{v & 0xFFFF:04x}"

# ---------------------------------------------------------------------------
# Compute PLA coefficients
# ---------------------------------------------------------------------------
slopes     = []
intercepts = []

for i in range(NSEG):
    x0 = XMIN + i * h
    x1 = x0 + h

    exp_x0 = math.exp(x0)
    exp_x1 = math.exp(x1)

    # Linear approximation: y = w * (x - x0) + exp(x0)
    # slope w = (exp(x1) - exp(x0)) / h
    # intercept b = exp(x0)  (value at segment start)
    w = (exp_x1 - exp_x0) / (x1 - x0)
    b = exp_x0

    slopes.append(w)
    intercepts.append(b)

# ---------------------------------------------------------------------------
# Write hex files
# ---------------------------------------------------------------------------
slopes_hex     = [float_to_uq115(w) for w in slopes]
intercepts_hex = [float_to_uq115(b) for b in intercepts]

slopes_path = os.path.join(OUTPUT_DIR, "pla_slopes.hex")
intercepts_path = os.path.join(OUTPUT_DIR, "pla_intercepts.hex")

with open(slopes_path, 'w') as f:
    for v in slopes_hex:
        f.write(hex16(v) + "\n")

with open(intercepts_path, 'w') as f:
    for v in intercepts_hex:
        f.write(hex16(v) + "\n")

# ---------------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------------
print(f"PLA ROM Generation Complete")
print(f"  Domain:        [{XMIN}, {XMAX}]")
print(f"  Segments:      {NSEG}")
print(f"  Segment width: h = {h}")
print(f"  Coefficient Q: Q1.{Q} ({COEFF_W}-bit unsigned)")
print(f"  Output files:  {slopes_path}")
print(f"                 {intercepts_path}")
print()
print(f"{'Seg':>3s} | {'x_start':>10s} | {'slope':>12s} | {'intercept':>12s} | {'w_hex':>6s} | {'b_hex':>6s}")
print("-" * 70)
for i in range(NSEG):
    x0 = XMIN + i * h
    print(f"{i:3d} | {x0:10.4f} | {slopes[i]:12.8f} | {intercepts[i]:12.8f} | {hex16(slopes_hex[i]):>6s} | {hex16(intercepts_hex[i]):>6s}")
