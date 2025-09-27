import numpy as np

def tile_matrix_A(matrix):
    """
    Tile matrix A vertically into 4x2 blocks for weight stationary systolic array
    For 4x4 matrix A, creates 2 tiles of size 4x2 each
    """
    rows, cols = matrix.shape
    tiles = []
    
    # Split vertically (column-wise) into 4x2 blocks
    for j in range(0, cols, 2):  # Step through columns by 2
        tile = matrix[:, j:j+2]  # Take all rows, 2 columns
        tiles.append(tile)
    
    return tiles

def tile_matrix_B(matrix):
    """
    Tile matrix B into 2x2 blocks for weight stationary systolic array
    For 4x4 matrix B, creates 4 tiles of size 2x2 each
    """
    rows, cols = matrix.shape
    tiles = []
    
    # Split into 2x2 blocks
    for i in range(0, rows, 2):      # Step through rows by 2
        tile_row = []
        for j in range(0, cols, 2):  # Step through cols by 2
            tile = matrix[i:i+2, j:j+2]  # Extract 2x2 block
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

def weight_stationary_systolic_multiply(A, B):
    """
    Simulate weight stationary systolic array matrix multiplication with correct tiling
    
    Matrix A: 4x4 -> tiled vertically into two 4x2 blocks
    Matrix B: 4x4 -> tiled into four 2x2 blocks
    
    Args:
        A: Input matrix (4x4)
        B: Weight matrix (4x4) 
    
    Returns:
        C: Result matrix
        computation_steps: List of computation steps for visualization
    """
    
    print(f"Original matrices:")
    print(f"Matrix A (4x4):\n{A}")
    print(f"Matrix B (4x4):\n{B}")
    print(f"Expected result (A @ B):\n{A @ B}")
    print("="*80)
    
    # Tile the matrices with correct scheme
    A_tiles = tile_matrix_A(A)  # Results in 2 tiles of 4x2 each
    B_tiles = tile_matrix_B(B)  # Results in 2x2 array of 2x2 tiles
    
    print(f"Tiling scheme:")
    print(f"Matrix A -> {len(A_tiles)} vertical tiles of shape {A_tiles[0].shape} (4x2)")
    print(f"Matrix B -> {len(B_tiles)}x{len(B_tiles[0])} tiles of shape {B_tiles[0][0].shape} (2x2)")
    print()
    
    # Show the tiles
    print("A_tiles:")
    for i, tile in enumerate(A_tiles):
        print(f"A_tile[{i}] (4x2):\n{tile}")
        print()
    
    print("B_tiles:")
    for i in range(len(B_tiles)):
        for j in range(len(B_tiles[0])):
            print(f"B_tile[{i}][{j}] (2x2):\n{B_tiles[i][j]}")
            print()
    
    # Initialize result matrix
    C = np.zeros_like(A)
    
    computation_steps = []
    
    print("="*80)
    print("SYSTOLIC ARRAY COMPUTATION:")
    print("="*80)
    
    # The result C will have the shape 4x4, computed by accumulating partial results
    # We need to think of this differently - the full matrix C is computed by:
    # C = A_tile[0] @ [B_tile[0][0], B_tile[0][1]] + A_tile[1] @ [B_tile[1][0], B_tile[1][1]]
    
    print("\nComputing full result matrix C:")
    print("C = A_tiles[0] @ [B_tiles[0][0], B_tiles[0][1]] + A_tiles[1] @ [B_tiles[1][0], B_tiles[1][1]]")
    
    # First partial result: A_tiles[0] multiplied with top row of B tiles
    print(f"\n{'='*50}")
    print("STEP 1: A_tiles[0] @ [B_tiles[0][0], B_tiles[0][1]]")
    print(f"{'='*50}")
    
    # Reconstruct top row of B from tiles
    B_top_row = np.hstack([B_tiles[0][0], B_tiles[0][1]])  # Combine B[0][0] and B[0][1]
    print(f"Reconstructed B top row (2x4): \n{B_top_row}")
    print(f"A_tiles[0] (4x2): \n{A_tiles[0]}")
    
    partial_result_1 = np.dot(A_tiles[0], B_top_row)
    print(f"A_tiles[0] @ B_top_row = (4x2) @ (2x4) = (4x4)")
    print(f"First partial result:\n{partial_result_1}")
    
    # Second partial result: A_tiles[1] multiplied with bottom row of B tiles  
    print(f"\n{'='*50}")
    print("STEP 2: A_tiles[1] @ [B_tiles[1][0], B_tiles[1][1]]")
    print(f"{'='*50}")
    
    # Reconstruct bottom row of B from tiles
    B_bottom_row = np.hstack([B_tiles[1][0], B_tiles[1][1]])  # Combine B[1][0] and B[1][1]
    print(f"Reconstructed B bottom row (2x4): \n{B_bottom_row}")
    print(f"A_tiles[1] (4x2): \n{A_tiles[1]}")
    
    partial_result_2 = np.dot(A_tiles[1], B_bottom_row)
    print(f"A_tiles[1] @ B_bottom_row = (4x2) @ (2x4) = (4x4)")
    print(f"Second partial result:\n{partial_result_2}")
    
    # Final result
    print(f"\n{'='*50}")
    print("FINAL STEP: Add partial results")
    print(f"{'='*50}")
    
    C = partial_result_1 + partial_result_2
    print(f"Final result C = partial_result_1 + partial_result_2:")
    print(f"C =\n{C}")
    
    # Now let's also show the tile-by-tile computation for understanding
    print(f"\n{'='*60}")
    print("TILE-BY-TILE BREAKDOWN (for understanding):")
    print(f"{'='*60}")
    
    for i in range(len(B_tiles)):        # Row of output tiles (0,1)
        for j in range(len(B_tiles[0])):  # Column of output tiles (0,1)
            
            print(f"\n--- Computing contribution to C region [{i*2}:{i*2+2}, {j*2}:{j*2+2}] ---")
            
            partial_sum = None
            
            # For weight stationary: iterate through A_tiles and corresponding B_tiles
            for k in range(len(A_tiles)):  # k=0,1 for the two A_tiles
                
                A_block = A_tiles[k]        # 4x2 block
                B_block = B_tiles[k][j]     # 2x2 block at position [k][j]
                
                print(f"\nContribution {k+1}: A_tiles[{k}] @ B_tiles[{k}][{j}]")
                print(f"A_tiles[{k}] (4x2):\n{A_block}")
                print(f"B_tiles[{k}][{j}] (2x2):\n{B_block}")
                
                # This gives us partial contribution to the full C matrix
                contribution = np.dot(A_block, B_block)
                print(f"Contribution (4x2): \n{contribution}")
                
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
            
            print(f"Final contribution to C region [{i*2}:{i*2+2}, {j*2}:{j*2+2}]: \n{partial_sum}")
            print(f"This corresponds to columns {j*2}:{j*2+2} of the final C matrix")
    
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
    # Create test 4x4 matrices
    np.random.seed(42)  # For reproducible results
    A = np.random.randint(1, 5, (4, 4))
    B = np.random.randint(1, 5, (4, 4))
    
    print("WEIGHT STATIONARY SYSTOLIC ARRAY - CORRECT TILING")
    print("="*80)
    
    # Perform systolic multiplication
    result_C, steps = weight_stationary_systolic_multiply(A, B)
    
    print(f"\n" + "="*80)
    print("FINAL VERIFICATION:")
    print("="*80)
    print(f"Systolic result:\n{result_C}")
    print(f"NumPy verification:\n{A @ B}")
    print(f"Results match: {np.allclose(result_C, A @ B)}")
    
    # Show the specific example (commented out)
    # demonstrate_specific_example()
    
    print(f"\n" + "="*80)
    print("SUMMARY:")
    print("="*80)
    print(f"- Matrix A (4x4) tiled into: 2 vertical tiles of 4x2")
    print(f"- Matrix B (4x4) tiled into: 4 tiles of 2x2 (2x2 arrangement)")
    print(f"- Matrix C (4x4) has same tiling as B: 4 tiles of 2x2")
    print(f"- Each C tile computed by: Σ(A_tile[k] @ B_tile[k][j]) for k=0,1")
    print(f"- Systolic array type: Weight Stationary")
    print(f"- Results verified: ✓" if np.allclose(result_C, A @ B) else "✗")