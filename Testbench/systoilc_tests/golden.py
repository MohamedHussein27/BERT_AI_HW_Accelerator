import numpy as np

num_rows = 512
num_cols = 32


A = np.zeros((num_rows, num_cols), dtype=np.int8)
B = np.ones((32, 32), dtype=np.int8)

pp = np.int8(0)
for r in range(num_rows):
    pp = np.int8(pp + np.int8(1)) 
    A[r, :] = pp

# Accumulate in int32
C = A.astype(np.int32) @ B.astype(np.int32)

# 3 tiles accumulated (first_iteration + 2 partial sums)
result = C * 3

with open("golden_model_result.txt", "w") as f:
    for row in result:
        f.write(" ".join(str(int(x)) for x in row) + "\n")

print("Done! Output saved to golden_model_result.txt")