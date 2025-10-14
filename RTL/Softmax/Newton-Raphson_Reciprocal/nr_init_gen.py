# nr_init_gen.py
import numpy as np

# config (tune)
Q = 26
WIDTH = 32
LUT_BITS = 4   # must match RTL LUT_BITS
LUT_SIZE = 1 << LUT_BITS

# expected input domain for 'a' (reciprocal). Softmax sums are positive.
A_MIN = 1e-6    # avoid zero
A_MAX = 64.0    # expected maximum sum you might see; tune to application (e.g., QKt 512*some max)
# If you expect sums up to e.g. 512, set A_MAX=512.

def to_q(x):
    # signed 32-bit Q
    val = int(round(x * (1 << Q)))
    return val & 0xFFFFFFFF

# Choose representative points for bins by splitting [A_MIN, A_MAX] uniformly in log-domain
# Using log spacing often gives better initial approximations across magnitudes.
bins = np.geomspace(A_MIN + 1e-12, A_MAX, LUT_SIZE+1)  # edges
centers = np.sqrt(bins[:-1] * bins[1:])  # geometric midpoints

vals = 1.0 / centers

with open('nr_init.hex','w') as f:
    for v in vals:
        f.write("{:08x}\n".format(to_q(v)))

print("Wrote nr_init.hex with", LUT_SIZE, "entries")
print("Sample centers (float) and q hex:")
for i in range(min(LUT_SIZE,8)):
    print(i, centers[i], vals[i], "{:08x}".format(to_q(vals[i])))
