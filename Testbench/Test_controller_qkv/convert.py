import os

def hex_to_signed_32bit(hex_str):
    """Converts an 8-character hex string to a 32-bit signed integer."""
    # If the hex string contains an unknown Verilog state, return "X"
    if 'x' in hex_str.lower():
        return "X"
    
    val = int(hex_str, 16)
    # Apply two's complement for 32-bit negative numbers
    if val >= 0x80000000:
        val -= 0x100000000
    return val

def parse_partial_sums(input_file, output_file):
    if not os.path.exists(input_file):
        print(f"Error: Could not find {input_file}")
        return

    parsed_rows = []

    # 1. Read and clean the .mem file
    with open(input_file, 'r') as f:
        for line in f:
            # Remove Verilog comments
            line = line.split('//')[0].strip()
            if not line:
                continue
            
            # Split by whitespace to separate addresses from data
            tokens = line.split()
            for token in tokens:
                # Ignore address markers (e.g., @0, @21a)
                if token.startswith('@'):
                    continue
                
                # Clean any formatting underscores
                token = token.replace('_', '')
                
                # Ensure the token is exactly 256 hex characters (1024 bits) long
                # Questa sometimes drops leading zeros; zfill restores them.
                if len(token) < 256:
                    token = token.zfill(256)
                    
                # 2. Extract thirty-two 32-bit numbers
                row_values = []
                
                # Slicing Right-to-Left: 
                # i=0 extracts the last 8 characters (bits 31:0)
                # i=31 extracts the first 8 characters (bits 1023:992)
                for i in range(32):
                    end_idx = 256 - (i * 8)
                    start_idx = end_idx - 8
                    chunk = token[start_idx:end_idx]
                    
                    val = hex_to_signed_32bit(chunk)
                    row_values.append(val)
                
                parsed_rows.append(row_values)

    # 3. Write to the output text file
    with open(output_file, 'w') as f_out:
        for row in parsed_rows:
            # Convert values to strings and join with a space
            row_str = " ".join(str(v) for v in row)
            f_out.write(row_str + "\n")
            
    print(f"Success! Processed {len(parsed_rows)} memory words.")
    print(f"Each word was split into 32 decimal numbers.")
    print(f"Data saved to {output_file}")

if __name__ == "__main__":
    # Ensure partial_sum.mem is in the same folder as this script, or provide the full path
    INPUT_MEM_FILE = "C:/integration_test/qkv_buffer.mem"
    OUTPUT_TXT_FILE = "C:/integration_test/txt_files/qkv_buffer.txt"
    
    parse_partial_sums(INPUT_MEM_FILE, OUTPUT_TXT_FILE)