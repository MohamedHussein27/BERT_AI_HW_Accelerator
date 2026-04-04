import os

# Hardware precision parameters
FRAC_BITS = 26
SCALE = 2 ** FRAC_BITS

def float_to_q5_26_hex(val):
    # Scale float to fixed-point integer and round
    q_val = int(round(val * SCALE))
    # Apply 32-bit Two's Complement mask for negative numbers
    q_val = q_val & 0xFFFFFFFF
    # Format as 8-character uppercase Hexadecimal string
    return f"{q_val:08X}"

def main():
    file_data = "D:\BERT_Local_SW\LN_sample_0_layer_0.txt"
    file_weights = "D:\BERT_Local_SW\layernorm_gamma_beta.txt"
    
    # ---------------------------------------------------------
    # 1. PARSE INPUTS AND EXPECTED OUTPUTS
    # ---------------------------------------------------------
    if not os.path.exists(file_data):
        print(f"Error: Could not find {file_data}")
        return

    all_rows = []
    print(f"Parsing data from {file_data}...")
    with open(file_data, "r") as f:
        for line in f:
            if line.startswith("row"):
                parts = line.split(":")
                if len(parts) == 2:
                    floats = [float(x) for x in parts[1].split()]
                    all_rows.append(floats)

    inputs = all_rows[0:128]
    expected = all_rows[128:256]

    print(" -> Writing inputs.hex...")
    with open("inputs.hex", "w") as f:
        for row in inputs:
            for val in row:
                f.write(float_to_q5_26_hex(val) + "\n")

    print(" -> Writing expected.hex...")
    with open("expected.hex", "w") as f:
        for row in expected:
            for val in row:
                f.write(float_to_q5_26_hex(val) + "\n")

    # ---------------------------------------------------------
    # 2. PARSE GAMMAS (WEIGHTS) AND BETAS (BIASES)
    # ---------------------------------------------------------
    if not os.path.exists(file_weights):
        print(f"Error: Could not find {file_weights}")
        return

    gammas = []
    betas = []
    current_array = None
    
    print(f"\nParsing weights from {file_weights}...")
    with open(file_weights, "r") as f:
        for line in f:
            line_stripped = line.strip()
            
            if "[LN1_gamma]" in line:
                current_array = gammas
            elif "[LN1_beta]" in line:
                current_array = betas
                
            elif line_stripped.startswith("[") and "]" in line_stripped:
                try:
                    val_str = line_stripped.split("]")[1].strip()
                    val = float(val_str)
                    if current_array is not None:
                        current_array.append(val)
                except ValueError:
                    pass

    print(" -> Writing real_gamma.hex...")
    with open("real_gamma.hex", "w") as f:
        for val in gammas:
            f.write(float_to_q5_26_hex(val) + "\n")

    print(" -> Writing real_beta.hex...")
    with open("real_beta.hex", "w") as f:
        for val in betas:
            f.write(float_to_q5_26_hex(val) + "\n")

    print("\nSUCCESS! All 4 Hex files generated successfully.")

if __name__ == "__main__":
    main()