import sys

def load_buffer(filename="output_buffer.txt"):
    data = []
    with open(filename, 'r') as f:
        for line in f:
            line = line.strip()
            if line.startswith("Row") and ":" in line:
                try:
                    nums = line.split(":", 1)[1].strip().split()
                    row = [int(x) for x in nums]
                    if len(row) == 32:
                        data.append(row)
                except:
                    pass
    return data

def deskew_and_diagnose(data, N=32, expected_per_mac=64):
    if not data:
        print("❌ No data loaded from output_buffer.txt!")
        return

    num_raw_rows = len(data)
    num_deskewed_rows = num_raw_rows - N + 1

    print(f"✅ Loaded {num_raw_rows} raw rows from RTL simulation")
    print(f"   Producing {num_deskewed_rows} deskewed (logical) rows\n")

    deskewed = []
    for r in range(num_deskewed_rows):
        row = []
        for c in range(N):
            src_row = r + c
            if src_row < num_raw_rows:
                val = data[src_row][c]
                row.append(val)
            else:
                row.append(0)
        deskewed.append(row)

    # === SAVE CLEAN OUTPUT ===
    with open("deskewed_output.txt", "w") as f:
        for i, row in enumerate(deskewed):
            f.write(f"Row {i}: {' '.join(map(str, row))}\n")

    print(f"deskewed finished ")
if __name__ == "__main__":
    filename = "output_buffer.txt"
    raw_data = load_buffer(filename)
    deskew_and_diagnose(raw_data)