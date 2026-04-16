# golden_quantized.py

# ============================================================
# Global quantization parameters
# ============================================================
GLOBAL_M = 1972133394   # 32'd1972133394
GLOBAL_S = 11           # 8'sd11

ACCUM_FACTOR = 24

ROWS_A = 512
COLS_A = 32
ROWS_B = 32
COLS_B = 32


# ============================================================
# Helpers
# ============================================================
def to_int8(x):
    x = int(x) & 0xFF
    return x - 256 if (x & 0x80) else x


def to_int32(x):
    x = int(x) & 0xFFFFFFFF
    return x - 2**32 if (x & 0x80000000) else x


def quantize_sv(data_in, scale_M, scale_S):
    data_in = int(data_in)
    scale_M = int(scale_M) & 0xFFFFFFFF   # unsigned 32-bit
    scale_S = int(scale_S)

    MUL_WIDTH = 64  # 32 + 32

    # -------------------------
    # Stage 1: Multiply (64-bit wrap)
    # -------------------------
    mul_reg = (data_in * scale_M) & ((1 << MUL_WIDTH) - 1)

    # convert to signed 64-bit
    if mul_reg & (1 << (MUL_WIDTH - 1)):
        mul_reg -= (1 << MUL_WIDTH)

    # -------------------------
    # Stage 2: Shift + Rounding
    # -------------------------
    rounding = (1 << (30 + scale_S)) & ((1 << MUL_WIDTH) - 1)

    shifted = mul_reg + rounding
    shifted = shifted >> (31 + scale_S)  # arithmetic shift

    # -------------------------
    # Stage 3: Clamp to int8
    # -------------------------
    if shifted > 127:
        return 127
    elif shifted < -128:
        return -128
    else:
        return shifted


# ============================================================
# Build matrix_32x32
# ============================================================
matrix_32x32 = [[0 for _ in range(COLS_B)] for _ in range(ROWS_B)]

for row in range(32):
    k = to_int8(row)
    for col in range(32):
        matrix_32x32[row][col] = k


# ============================================================
# Build matrix_512x32
# Matching your RTL:
#
#   k = row;
#   if (row >= 256) k = 6;
#   if (row == 511)  k = 1;
#   matrix_512x32[row] = {32{k+2}};
# ============================================================
matrix_512x32 = [[0 for _ in range(COLS_A)] for _ in range(ROWS_A)]

for row in range(512):
    k = to_int8(row)

    if row >= 256:
        k = 6

    if row == 511:
        k = 1

    val = to_int8(k + 2)

    for col in range(32):
        matrix_512x32[row][col] = val


# ============================================================
# Matrix multiplication: (512x32) x (32x32) -> (512x32)
# Then multiply by 24
# ============================================================
result_int32 = [[0 for _ in range(COLS_B)] for _ in range(ROWS_A)]

for i in range(ROWS_A):
    for j in range(COLS_B):
        acc = 0
        for k in range(COLS_A):
            a = matrix_512x32[i][k]
            b = matrix_32x32[k][j]
            acc += a * b

        acc = to_int32(acc)
        acc = to_int32(acc * ACCUM_FACTOR)

        result_int32[i][j] = acc


# ============================================================
# Quantize int32 output to int8 using the SV model
# ============================================================
result_q8 = [[0 for _ in range(COLS_B)] for _ in range(ROWS_A)]

for i in range(ROWS_A):
    for j in range(COLS_B):
        result_q8[i][j] = quantize_sv(result_int32[i][j], GLOBAL_M, GLOBAL_S)


# ============================================================
# Save outputs
# ============================================================
with open("golden_output_int32.txt", "w") as f:
    for row in result_int32:
        f.write(" ".join(str(int(x)) for x in row) + "\n")

with open("golden_output_q8.txt", "w") as f:
    for row in result_q8:
        f.write(" ".join(str(int(x)) for x in row) + "\n")

print("Generated:")
print("  golden_output_int32.txt")
print("  golden_output_q8.txt")