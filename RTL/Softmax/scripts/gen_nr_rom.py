#!/usr/bin/env python3
"""
gen_nr_rom.py — Generate Newton-Raphson reciprocal initial guess LUT.

The reciprocal module normalizes the input to [1.0, 2.0) before NR iteration.
The LUT is indexed by the top LUT_BITS of the fractional part within [1.0, 2.0).

For the normalized value a_norm in [1.0, 2.0):
  index i covers: a_norm in [1 + i/16, 1 + (i+1)/16)
  midpoint = 1 + (i + 0.5) / 16
  guess = 1 / midpoint

Output: 16 entries in Q8.24 unsigned (32-bit), stored as 8-digit hex.
"""

import math
import os

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
WIDTH    = 32
Q        = 24           # Q8.24
LUT_BITS = 4
LUT_SIZE = 1 << LUT_BITS  # 16 entries

OUTPUT_DIR = os.path.dirname(os.path.abspath(__file__))

# ---------------------------------------------------------------------------
# Generate LUT
# ---------------------------------------------------------------------------
def float_to_q824(val):
    """Convert float to Q8.24 unsigned 32-bit."""
    raw = int(round(val * (1 << Q)))
    if raw < 0:
        raw = 0
    if raw > (1 << WIDTH) - 1:
        raw = (1 << WIDTH) - 1
    return raw

def hex32(v):
    return f"{v & 0xFFFFFFFF:08x}"

# LUT subdivides [1.0, 2.0) into 16 sub-intervals
# Each interval covers width = 1.0/16 = 0.0625
lut_entries = []
for i in range(LUT_SIZE):
    a_lo  = 1.0 + i / LUT_SIZE
    a_hi  = 1.0 + (i + 1) / LUT_SIZE
    a_mid = 1.0 + (i + 0.5) / LUT_SIZE
    guess = 1.0 / a_mid
    lut_entries.append((a_lo, a_hi, a_mid, guess))

# ---------------------------------------------------------------------------
# Write hex file
# ---------------------------------------------------------------------------
output_path = os.path.join(OUTPUT_DIR, "nr_init.hex")
with open(output_path, 'w') as f:
    for _, _, _, guess in lut_entries:
        f.write(hex32(float_to_q824(guess)) + "\n")

# ---------------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------------
print(f"Newton-Raphson LUT Generation Complete")
print(f"  Width:       {WIDTH}-bit Q8.{Q}")
print(f"  LUT entries: {LUT_SIZE}")
print(f"  Range:       [1.0, 2.0) normalized")
print(f"  Output file: {output_path}")
print()
print(f"{'Idx':>3s} | {'a_range':>20s} | {'a_mid':>10s} | {'1/a_mid':>12s} | {'hex':>10s}")
print("-" * 65)
for i, (a_lo, a_hi, a_mid, guess) in enumerate(lut_entries):
    print(f"{i:3d} | [{a_lo:8.5f}, {a_hi:8.5f}) | {a_mid:10.6f} | {guess:12.8f} | {hex32(float_to_q824(guess)):>10s}")
