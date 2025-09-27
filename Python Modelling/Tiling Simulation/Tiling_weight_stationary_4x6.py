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
    # Test with 4x6 and 6x6 matrices using 2x2 systolic array
    print("WEIGHT STATIONARY SYSTOLIC ARRAY - 4x6 @ 6x6 MATRICES")
    print("="*80)
    
    # Create test matrices
    np.random.seed(42)  # For reproducible results
    A = np.random.randint(1, 5, (4, 6))  # 4x6 matrix
    B = np.random.randint(1, 5, (6, 6))  # 6x6 matrix
    
    # Perform systolic multiplication with 2x2 systolic array
    result_C, steps = weight_stationary_systolic_multiply(A, B, systolic_height=2, systolic_width=2)
    
    print(f"\n" + "="*80)
    print("FINAL VERIFICATION:")
    print("="*80)
    print(f"Systolic result ({result_C.shape}):\n{result_C}")
    print(f"NumPy verification ({(A @ B).shape}):\n{A @ B}")
    print(f"Results match: {np.allclose(result_C, A @ B)}")
    
    # Show the specific example (commented out)
    # demonstrate_specific_example()
    
    print(f"\n" + "="*80)
    print("SUMMARY:")
    print("="*80)
    print(f"- Matrix A ({A.shape[0]}×{A.shape[1]}) tiled into: {A.shape[1]//2} vertical tiles of {A.shape[0]}×2")
    print(f"- Matrix B ({B.shape[0]}×{B.shape[1]}) tiled into: {B.shape[0]//2}×{B.shape[1]//2} tiles of 2×2")
    print(f"- Matrix C ({result_C.shape[0]}×{result_C.shape[1]}) computed through systolic processing")
    print(f"- Systolic array size: 2×2")
    print(f"- Total computation steps: {len(steps)}")
    print(f"- Systolic array type: Weight Stationary")
    print(f"- Results verified: ✓" if np.allclose(result_C, A @ B) else "✗")
    
    print(f"\n" + "="*60)
    print("TILING BREAKDOWN:")
    print("="*60)
    print(f"A_tiles: {A.shape[1]//2} tiles (vertical split)")
    for i in range(A.shape[1]//2):
        print(f"  A_tile[{i}]: shape {A.shape[0]}×2, covers columns {i*2}-{i*2+1}")
    print(f"B_tiles: {B.shape[0]//2}×{B.shape[1]//2} tiles arrangement")
    for i in range(B.shape[0]//2):
        for j in range(B.shape[1]//2):
            print(f"  B_tile[{i}][{j}]: shape 2×2, covers rows {i*2}-{i*2+1}, cols {j*2}-{j*2+1}")
    
    print(f"\nComputation pattern:")
    print(f"For each output column group j:")
    print(f"  C[:, {2}*j:{2}*(j+1)] = Σ(k=0 to {A.shape[1]//2-1}) A_tile[k] @ B_tile[k][j]")