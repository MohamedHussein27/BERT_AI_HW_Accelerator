import numpy as np

def tile_matrix_A(matrix, systolic_height=2):
    """
    Tile matrix A vertically for weight stationary systolic array
    Tiles are sized to match systolic array height × systolic_width
    
    For matrix A (M×K), creates K/systolic_width tiles of size M×systolic_width each
    """
    rows, cols = matrix.shape
    tiles = []
    
    # Split vertically (column-wise) into M×systolic_width blocks
    for j in range(0, cols, systolic_height):  # Step through columns by systolic_width
        tile = matrix[:, j:j+systolic_height]  # Take all rows, systolic_width columns
        tiles.append(tile)
    
    return tiles

def tile_matrix_B(matrix, systolic_height=2, systolic_width=2):
    """
    Tile matrix B into blocks sized for weight stationary systolic array
    
    For matrix B (K×N), creates tiles of size systolic_height×systolic_width
    """
    rows, cols = matrix.shape
    tiles = []
    
    # Split into systolic_height×systolic_width blocks
    for i in range(0, rows, systolic_height):      # Step through rows by systolic_height
        tile_row = []
        for j in range(0, cols, systolic_width):   # Step through cols by systolic_width
            tile = matrix[i:i+systolic_height, j:j+systolic_width]  # Extract block
            tile_row.append(tile)
        tiles.append(tile_row)
    
    return tiles

def systolic_block_multiply(A_block, B_block, partial_sum=None):
    """
    Simulate systolic array block multiplication
    A_block: input activation block (4x2)
    B_block: weight block (2x2) - stationary in systolic array
    partial_sum: previous partial sum to accumulate
    """
    result = np.dot(A_block, B_block)
    
    if partial_sum is not None:
        result += partial_sum
    
    return result

def weight_stationary_systolic_multiply(A, B, systolic_height=2, systolic_width=2):
    """
    Simulate weight stationary systolic array matrix multiplication with configurable tiling
    
    Matrix A: M×K -> tiled vertically into K/systolic_width tiles of size M×systolic_width
    Matrix B: K×N -> tiled into (K/systolic_height)×(N/systolic_width) tiles of size systolic_height×systolic_width
    
    Args:
        A: Input matrix (M×K)
        B: Weight matrix (K×N) 
        systolic_height: Height of systolic array (default 2)
        systolic_width: Width of systolic array (default 2)
    
    Returns:
        C: Result matrix
        computation_steps: List of computation steps for visualization
    """
    
    print(f"Original matrices:")
    print(f"Matrix A ({A.shape[0]}×{A.shape[1]}):\n{A}")
    print(f"Matrix B ({B.shape[0]}×{B.shape[1]}):\n{B}")
    print(f"Expected result (A @ B) ({A.shape[0]}×{B.shape[1]}):\n{A @ B}")
    print("="*80)
    
    # Tile the matrices with correct scheme
    A_tiles = tile_matrix_A(A, systolic_height)  # Results in K/systolic_width tiles of M×systolic_width each
    B_tiles = tile_matrix_B(B, systolic_height, systolic_width)  # Results in array of systolic_height×systolic_width tiles
    
    print(f"Tiling scheme for {systolic_height}×{systolic_width} systolic array:")
    print(f"Matrix A -> {len(A_tiles)} vertical tiles of shape {A_tiles[0].shape}")
    print(f"Matrix B -> {len(B_tiles)}×{len(B_tiles[0])} tiles of shape {B_tiles[0][0].shape}")
    print()
    
    # Show the tiles
    print("A_tiles:")
    for i, tile in enumerate(A_tiles):
        print(f"A_tile[{i}] {tile.shape}:\n{tile}")
        print()
    
    print("B_tiles:")
    for i in range(len(B_tiles)):
        for j in range(len(B_tiles[0])):
            print(f"B_tile[{i}][{j}] {B_tiles[i][j].shape}:\n{B_tiles[i][j]}")
            print()
    
    # Initialize result matrix
    C = np.zeros((A.shape[0], B.shape[1]))
    
    computation_steps = []
    
    print("="*80)
    print("SYSTOLIC ARRAY COMPUTATION:")
    print("="*80)
    
    # The result C will have the shape M×N, computed by accumulating partial results
    print(f"\nComputing full result matrix C ({A.shape[0]}×{B.shape[1]}):")
    print("C = sum over k of: A_tiles[k] @ [B_tiles[k][0], B_tiles[k][1], ...]")
    
    # Compute the result by processing each "row" of B tiles
    C_partial_results = []
    
    for k in range(len(A_tiles)):  # For each A_tile and corresponding B_tile row
        print(f"\n{'='*50}")
        print(f"STEP {k+1}: A_tiles[{k}] @ B_tiles[{k}][:]")
        print(f"{'='*50}")
        
        # Reconstruct the k-th "row" of B tiles
        B_row_tiles = B_tiles[k]  # Get the k-th row of B tiles
        B_reconstructed_row = np.hstack(B_row_tiles)  # Combine tiles horizontally
        
        print(f"Reconstructed B row {k} ({B_reconstructed_row.shape}): \n{B_reconstructed_row}")
        print(f"A_tiles[{k}] ({A_tiles[k].shape}): \n{A_tiles[k]}")
        
        partial_result = np.dot(A_tiles[k], B_reconstructed_row)
        print(f"A_tiles[{k}] @ B_row_{k} = {A_tiles[k].shape} @ {B_reconstructed_row.shape} = {partial_result.shape}")
        print(f"Partial result {k+1}:\n{partial_result}")
        
        C_partial_results.append(partial_result)
    
    # Final result
    print(f"\n{'='*50}")
    print("FINAL STEP: Sum all partial results")
    print(f"{'='*50}")
    
    C = sum(C_partial_results)
    print(f"Final result C = sum of partial results:")
    print(f"C ({C.shape}):\n{C}")
    
    # Now let's also show the tile-by-tile computation for understanding
    print(f"\n{'='*60}")
    print("TILE-BY-TILE BREAKDOWN (for understanding):")
    print(f"{'='*60}")
    
    for i in range(len(B_tiles)):        # Row of B tiles
        for j in range(len(B_tiles[0])):  # Column of B tiles
            
            print(f"\n--- Computing contribution to C region [:, {j*systolic_width}:{j*systolic_width+systolic_width}] ---")
            
            partial_sum = None
            
            # For weight stationary: iterate through A_tiles and corresponding B_tiles
            for k in range(len(A_tiles)):  # k for each A_tile
                
                A_block = A_tiles[k]        # M×systolic_width block
                B_block = B_tiles[k][j]     # systolic_height×systolic_width block at position [k][j]
                
                print(f"\nContribution {k+1}: A_tiles[{k}] @ B_tiles[{k}][{j}]")
                print(f"A_tiles[{k}] {A_block.shape}:\n{A_block}")
                print(f"B_tiles[{k}][{j}] {B_block.shape}:\n{B_block}")
                
                # This gives us partial contribution to specific columns of the full C matrix
                contribution = np.dot(A_block, B_block)
                print(f"Contribution {contribution.shape}: \n{contribution}")
                
                if partial_sum is None:
                    partial_sum = contribution
                else:
                    partial_sum += contribution
                    print(f"Accumulated contribution: \n{partial_sum}")
                
                step_info = {
                    'output_region': (i, j),
                    'step': k + 1,
                    'A_block': A_block.copy(),
                    'B_block': B_block.copy(),
                    'contribution': contribution.copy(),
                    'is_final': k == len(A_tiles) - 1
                }
                computation_steps.append(step_info)
            
            print(f"Final contribution to C columns {j*systolic_width}:{j*systolic_width+systolic_width}: \n{partial_sum}")
            print(f"This corresponds to columns {j*systolic_width}-{j*systolic_width+systolic_width-1} of the final C matrix")
    
    return C, computation_steps

def demonstrate_specific_example():
    """
    Demonstrate the specific example mentioned in the user's correction
    """
    print("\n" + "="*80)
    print("DETAILED EXAMPLE - Computing C[0][0] and C[0][1]")
    print("="*80)
    
    # Create a simple example for clarity
    A = np.array([[1, 2, 3, 4],
                  [5, 6, 7, 8], 
                  [9, 10,11,12],
                  [13,14,15,16]])
    
    B = np.array([[1, 1, 2, 2],
                  [1, 1, 2, 2],
                  [3, 3, 4, 4],
                  [3, 3, 4, 4]])
    
    print(f"Example matrices:")
    print(f"A =\n{A}")
    print(f"B =\n{B}")
    
    # Tile them
    A_tiles = tile_matrix_A(A)
    B_tiles = tile_matrix_B(B)
    
    print(f"\nTiled A (vertical 4x2 tiles):")
    print(f"A_tiles[0] =\n{A_tiles[0]}")
    print(f"A_tiles[1] =\n{A_tiles[1]}")
    
    print(f"\nTiled B (2x2 tiles):")
    for i in range(2):
        for j in range(2):
            print(f"B_tiles[{i}][{j}] =\n{B_tiles[i][j]}")
    
    print(f"\n--- Computing C_tile[0][0] ---")
    print("Step 1: A_tiles[0] @ B_tiles[0][0]")
    partial1 = np.dot(A_tiles[0], B_tiles[0][0])
    print(f"A_tiles[0] @ B_tiles[0][0] =\n{A_tiles[0]} @\n{B_tiles[0][0]} =\n{partial1}")
    
    print("Step 2: A_tiles[1] @ B_tiles[1][0] + previous partial sum")
    partial2 = np.dot(A_tiles[1], B_tiles[1][0])
    final_c00 = partial1 + partial2
    print(f"A_tiles[1] @ B_tiles[1][0] =\n{A_tiles[1]} @\n{B_tiles[1][0]} =\n{partial2}")
    print(f"Final C_tile[0][0] = {partial1} + {partial2} =\n{final_c00}")
    
    print(f"\n--- Computing C_tile[0][1] ---")
    print("Step 1: A_tiles[0] @ B_tiles[0][1]")
    partial1 = np.dot(A_tiles[0], B_tiles[0][1])
    print(f"A_tiles[0] @ B_tiles[0][1] =\n{A_tiles[0]} @\n{B_tiles[0][1]} =\n{partial1}")
    
    print("Step 2: A_tiles[1] @ B_tiles[1][1] + previous partial sum")
    partial2 = np.dot(A_tiles[1], B_tiles[1][1])
    final_c01 = partial1 + partial2
    print(f"A_tiles[1] @ B_tiles[1][1] =\n{A_tiles[1]} @\n{B_tiles[1][1]} =\n{partial2}")
    print(f"Final C_tile[0][1] = {partial1} + {partial2} =\n{final_c01}")

# Test the implementation
if __name__ == "__main__":
    # Test with 512x768 and 768x768 matrices using 32x32 systolic array
    print("WEIGHT STATIONARY SYSTOLIC ARRAY - 512x768 @ 768x768 MATRICES")
    print("="*80)
    
    # Create test matrices
    print("Creating large test matrices (this may take a moment)...")
    np.random.seed(42)  # For reproducible results
    A = np.random.randint(1, 5, (512, 768))  # 512x768 matrix
    B = np.random.randint(1, 5, (768, 3072))  # 768x768 matrix
    
    print(f"Matrix A shape: {A.shape}")
    print(f"Matrix B shape: {B.shape}")
    print(f"Expected result shape: {A.shape[0]}x{B.shape[1]}")
    print(f"Systolic array size: 32x32")
    
    # Calculate expected tiling
    systolic_height, systolic_width = 32, 32
    num_A_tiles = A.shape[1] // systolic_width  # 768 // 32 = 24
    num_B_tile_rows = B.shape[0] // systolic_height  # 768 // 32 = 24  
    num_B_tile_cols = B.shape[1] // systolic_width  # 768 // 32 = 24
    
    print(f"\nTiling breakdown:")
    print(f"- A_tiles: {num_A_tiles} vertical tiles of {A.shape[0]}x{systolic_width}")
    print(f"- B_tiles: {num_B_tile_rows}x{num_B_tile_cols} tiles of {systolic_height}x{systolic_width}")
    print(f"- Total B tiles: {num_B_tile_rows * num_B_tile_cols}")
    
    # For demonstration with large matrices, we'll show a simplified version
    # without printing all the intermediate matrices (too much output)
    print(f"\n" + "="*80)
    print("PERFORMING SYSTOLIC COMPUTATION...")
    print("="*80)
    
    # Tile the matrices
    A_tiles = tile_matrix_A(A, systolic_height)
    B_tiles = tile_matrix_B(B, systolic_height, systolic_width)
    
    print(f"Tiling complete:")
    print(f"- Created {len(A_tiles)} A_tiles, each of shape {A_tiles[0].shape}")
    print(f"- Created {len(B_tiles)}x{len(B_tiles[0])} B_tiles, each of shape {B_tiles[0][0].shape}")
    
    # Compute the result efficiently
    print(f"\nComputing matrix multiplication using systolic array simulation...")
    
    # Initialize result matrix
    C = np.zeros((A.shape[0], B.shape[1]))
    
    # Process each step of the systolic computation
    computation_count = 0
    total_computations = len(A_tiles) * len(B_tiles[0])
    
    for k in range(len(A_tiles)):  # For each A_tile
        print(f"Processing A_tile[{k}] ({k+1}/{len(A_tiles)})...")
        
        # Reconstruct the k-th "row" of B tiles  
        B_row_tiles = B_tiles[k]
        B_reconstructed_row = np.hstack(B_row_tiles)
        
        # Compute partial result
        partial_result = np.dot(A_tiles[k], B_reconstructed_row)
        
        # Add to final result
        C += partial_result
        
        computation_count += len(B_tiles[0])
        progress = (computation_count / total_computations) * 100
        print(f"  Progress: {progress:.1f}% complete")
    
    print(f"\nSystolic computation complete!")
    
    # Full verification against NumPy result
    print(f"\n" + "="*80)
    print("FULL VERIFICATION:")
    print("="*80)
    
    print(f"Computing NumPy reference result for comparison...")
    print(f"This may take a moment for large matrices...")
    
    # Compute the expected result using NumPy
    C_expected = A @ B
    
    print(f"Comparing systolic result with NumPy reference...")
    print(f"Systolic result shape: {C.shape}")
    print(f"NumPy result shape: {C_expected.shape}")
    
    # Check if results match
    results_match = np.allclose(C, C_expected, rtol=1e-10, atol=1e-12)
    
    if results_match:
        print("✓ FULL VERIFICATION PASSED!")
        print("  Systolic array result matches NumPy exactly")
    else:
        print("✗ VERIFICATION FAILED!")
        max_diff = np.max(np.abs(C - C_expected))
        mean_diff = np.mean(np.abs(C - C_expected))
        print(f"  Maximum absolute difference: {max_diff}")
        print(f"  Mean absolute difference: {mean_diff}")
        
        # Show some sample differences for debugging
        diff_matrix = np.abs(C - C_expected)
        print(f"  Top-left 5x5 absolute differences:")
        print(f"  {diff_matrix[:5, :5]}")
    
    print(f"\nResult verification: {'✓ PASSED' if results_match else '✗ FAILED'}")
    
    print(f"\n" + "="*80)
    print("PERFORMANCE SUMMARY:")
    print("="*80)
    print(f"- Matrix A: {A.shape[0]:,} × {A.shape[1]:,}")
    print(f"- Matrix B: {B.shape[0]:,} × {B.shape[1]:,}")
    print(f"- Result C: {C.shape[0]:,} × {C.shape[1]:,}")
    print(f"- Systolic array: {systolic_height} × {systolic_width}")
    print(f"- A tiles: {len(A_tiles):,} tiles of {A_tiles[0].shape}")
    print(f"- B tiles: {len(B_tiles):,} × {len(B_tiles[0]):,} = {len(B_tiles) * len(B_tiles[0]):,} total tiles")
    print(f"- Total systolic operations: {len(A_tiles) * len(B_tiles[0]):,}")
    print(f"- Elements per A tile: {A_tiles[0].size:,}")
    print(f"- Elements per B tile: {B_tiles[0][0].size:,}")
    
    print(f"\n" + "="*60)
    print("DETAILED TILING INFO:")
    print("="*60)
    print(f"Matrix A ({A.shape[0]}×{A.shape[1]}) → {len(A_tiles)} vertical tiles:")
    for i in range(min(3, len(A_tiles))):  # Show first 3 tiles
        start_col = i * systolic_width
        end_col = start_col + systolic_width
        print(f"  A_tile[{i}]: {A_tiles[i].shape}, covers columns {start_col}-{end_col-1}")
    if len(A_tiles) > 3:
        print(f"  ... and {len(A_tiles)-3} more tiles")
    
    print(f"\nMatrix B ({B.shape[0]}×{B.shape[1]}) → {len(B_tiles)}×{len(B_tiles[0])} tile grid:")
    tiles_shown = 0
    for i in range(min(3, len(B_tiles))):
        for j in range(min(3, len(B_tiles[0]))):
            start_row = i * systolic_height
            end_row = start_row + systolic_height
            start_col = j * systolic_width  
            end_col = start_col + systolic_width
            print(f"  B_tile[{i}][{j}]: {B_tiles[i][j].shape}, covers rows {start_row}-{end_row-1}, cols {start_col}-{end_col-1}")
            tiles_shown += 1
    total_b_tiles = len(B_tiles) * len(B_tiles[0])
    if total_b_tiles > tiles_shown:
        print(f"  ... and {total_b_tiles - tiles_shown} more tiles")
    
    print(f"\nComputation pattern for each output column group j:")
    print(f"  C[:, 32*j:32*(j+1)] = Σ(k=0 to {len(A_tiles)-1}) A_tile[k] @ B_tile[k][j]")
    print(f"  Total column groups: {len(B_tiles[0])}")
    
    # Optional: Show a small sample of the result
    print(f"\n" + "="*60)  
    print("RESULT SAMPLE (top-left 8x8):")
    print("="*60)
    print(C[:8, :8])
    
    print(f"\nSystemic array simulation completed successfully!")
    print(f"Result matrix C computed with shape {C.shape}")