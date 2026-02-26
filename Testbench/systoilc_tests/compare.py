def compare_files(golden_file, hardware_file):
    with open(golden_file, 'r') as gf, open(hardware_file, 'r') as hf:
        golden_lines = gf.readlines()
        hardware_lines = hf.readlines()

    if len(golden_lines) != len(hardware_lines):
        print(f"Warning: File length mismatch! Golden: {len(golden_lines)} lines, Hardware: {len(hardware_lines)} lines.")

    mismatches = 0
    total_elements = 0
    max_errors_to_print = 10

    # Iterate through the lines (using the shorter file length to avoid crashes)
    for row_idx in range(min(len(golden_lines), len(hardware_lines))):
        # 1. Parse Golden Line: Just split by spaces
        golden_vals = [int(x) for x in golden_lines[row_idx].split() if x.strip()]
        
        # 2. Parse Hardware Line: Remove "Row X: " prefix then split
        hw_line_raw = hardware_lines[row_idx]
        if ":" in hw_line_raw:
            hw_line_data = hw_line_raw.split(":")[1] # Get everything after the colon
        else:
            hw_line_data = hw_line_raw
        
        hw_vals = [int(x) for x in hw_line_data.split() if x.strip()]

        # 3. Compare row lengths
        if len(golden_vals) != len(hw_vals):
            print(f"Row {row_idx}: Width mismatch! Golden has {len(golden_vals)} elements, HW has {len(hw_vals)}")
            continue

        # 4. Compare element by element
        for col_idx in range(len(golden_vals)):
            total_elements += 1
            if golden_vals[col_idx] != hw_vals[col_idx]:
                mismatches += 1
                if mismatches <= max_errors_to_print:
                    print(f"Mismatch at Row {row_idx}, Col {col_idx}: Golden={golden_vals[col_idx]}, HW={hw_vals[col_idx]}")

    # Final Report
    if mismatches == 0:
        print("\nSUCCESS: All values match perfectly!")
    else:
        print(f"\nFAILURE: Found {mismatches} mismatches out of {total_elements} elements.")

# Run the comparison
compare_files("golden_model_result.txt", "deskewed_output.txt")