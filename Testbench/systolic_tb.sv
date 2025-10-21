module systolic_tb;

    parameter DATAWIDTH = 8;
    parameter N_SIZE = 2;
    parameter CLK_PERIOD = 10;

    logic clk;
    logic rst_n;
    logic wt_en;
    logic valid_in;
    logic [(DATAWIDTH) - 1:0] matrix_A [N_SIZE-1:0];
    logic [(DATAWIDTH*3) - 1:0] matrix_B [N_SIZE-1:0];
    logic [DATAWIDTH-1:0] wt_flat [N_SIZE*N_SIZE-1:0];
    logic [(DATAWIDTH*3) - 1:0] matrix_C [N_SIZE-1:0];

    // Clock generation
    initial begin
        clk = 0;
        forever #(CLK_PERIOD/2) clk = ~clk;
    end

    // DUT instantiation
    systolic #(
        .DATAWIDTH(DATAWIDTH),
        .N_SIZE(N_SIZE)
    ) dut (
        .clk(clk),
        .rst_n(rst_n),
        .wt_en(wt_en),
        .valid_in(valid_in),
        .matrix_A(matrix_A),
        .matrix_B(matrix_B),
        .wt_flat(wt_flat),
        .matrix_C(matrix_C)
    );

    // Input data - stored as columns (each column feeds one input row)
    logic [DATAWIDTH-1:0] A_tile1 [1:0][4:0];
    logic [DATAWIDTH-1:0] A_tile2 [1:0][4:0];

    logic [DATAWIDTH-1:0] W_tile1[1:0][1:0];
    logic [DATAWIDTH-1:0] W_tile2[1:0][1:0];
    logic [DATAWIDTH-1:0] W_tile3[1:0][1:0];
    logic [DATAWIDTH-1:0] W_tile4[1:0][1:0];

    // Storage for captured outputs (4 rows of output)
    logic [(DATAWIDTH*3) - 1:0] partial_sum [4:0][1:0] = '{default: 0};
    logic [(DATAWIDTH*3) - 1:0] result_matrix [3:0][3:0];

    // Initialize data
    initial begin
        // Weight matrix (stationary in PEs)
        W_tile1[0][0] = 1;  W_tile1[0][1] = 2;  W_tile3[0][0] = 3;  W_tile3[0][1] = 4;
        W_tile1[1][0] = 5;  W_tile1[1][1] = 6;  W_tile3[1][0] = 7;  W_tile3[1][1] = 8;
        W_tile2[0][0] = 9;  W_tile2[0][1] = 10; W_tile4[0][0] = 11; W_tile4[0][1] = 12;
        W_tile2[1][0] = 13; W_tile2[1][1] = 14; W_tile4[1][0] = 15; W_tile4[1][1] = 16;

        A_tile1[0][0] = 1; A_tile1[0][1] = 5;  A_tile1[0][2] = 9;  A_tile1[0][3] = 13;  A_tile1[0][4] = 0;
        A_tile1[1][0] = 0; A_tile1[1][1] = 2; A_tile1[1][2] = 6; A_tile1[1][3] = 10; A_tile1[1][4] = 14;

        A_tile2[0][0] = 3; A_tile2[0][1] = 7;  A_tile2[0][2] = 11;  A_tile2[0][3] = 15;  A_tile2[0][4] = 0;
        A_tile2[1][0] = 0; A_tile2[1][1] = 4; A_tile2[1][2] = 8; A_tile2[1][3] = 12; A_tile2[1][4] = 16;
    end

    // Task: flatten and load weights into wt_flat
    task automatic load_weights(input logic [DATAWIDTH-1:0] W_tile [1:0][1:0]);
        for (int i = 0; i < N_SIZE; i++) begin
            for (int j = 0; j < N_SIZE; j++) begin
                wt_flat[i*N_SIZE + j] = W_tile[i][j];
            end
        end
    endtask

    // Task: run one tile multiply (skewed input) and capture partial sums
    task automatic run_tile(
        input logic [DATAWIDTH-1:0] W_tile_in [1:0][1:0],
        input logic [DATAWIDTH-1:0] A_tile_in [1:0][4:0],
        input bit use_partial // if set, feed partial_sum into matrix_B
    );
        // load weights
        load_weights(W_tile_in);
        wt_en = 1;
        @(negedge clk);
        wt_en = 0;
        @(negedge clk);

        valid_in = 1;
        $display("\nFeeding skewed data (use_partial=%0d)...\n", use_partial);
        for (int cycle = 0; cycle < 5; cycle++) begin
            // Feed data to each input row from its sequence
            for (int row = 0; row < N_SIZE; row++) begin
                matrix_A[row] = A_tile_in[row][cycle];
            end

            // matrix_B either zeroed or loaded from partial_sum
            for (int col = 0; col < N_SIZE; col++) begin
                if (use_partial) begin
                    matrix_B[col] = partial_sum[cycle][col];
                end else begin
                    matrix_B[col] = 0;
                end
            end

            $display("Cycle %0d: A[0]=%2d A[1]=%2d", cycle, matrix_A[0], matrix_A[1]);

            @(negedge clk);

            // capture outputs when valid
            if (cycle >= 1 && cycle <= 3) begin
                int out_row = cycle - 1;
                partial_sum[out_row][0] = matrix_C[0];
                partial_sum[out_row][1] = matrix_C[1];
                $display("  --> Output row %0d captured: [%0d, %0d]", out_row, partial_sum[out_row][0], partial_sum[out_row][1]);
            end
        end

        // capture final output row
        partial_sum[3][0] = matrix_C[0];
        partial_sum[3][1] = matrix_C[1];
        $display("  --> Output row 3 captured: [%0d, %0d]", partial_sum[3][0], partial_sum[3][1]);

        // leave valid asserted for a cycle then deassert (keeps same timing as original)
        valid_in = 0;
        @(negedge clk);
    endtask

    // Main test
    initial begin
        $display("=== Systolic Array Test (Weight Stationary - Skewed Input) ===\n");
        // Initialize
        rst_n = 0;
        wt_en = 0;
        valid_in = 0;
        for (int i = 0; i < N_SIZE; i++) begin
            matrix_A[i] = 0;
            matrix_B[i] = 0;
        end

        repeat(2) @(negedge clk);
        rst_n = 1;
        @(negedge clk);

        // tile A1 * W1
        $display("tile A1 * tile W1");
        run_tile(W_tile1, A_tile1, 0);
        // tile A2 * W2 (use partial sums as B input)
        $display("tile A2 * tile W2");
        run_tile(W_tile2, A_tile2, 1);
        // move captured partial sums into result_matrix columns 0..1
        for (int i = 0; i < 4; i++) begin
            for (int j = 0; j < 2; j++) begin
                result_matrix[i][j] = partial_sum[i][j];
            end
        end

        // display an intermediate formatted block similar to original
        $display("| %0d %0d %0d %0d |\n| %0d %0d %0d %0d |\n| %0d %0d %0d %0d |\n| %0d %0d %0d %0d |",
             result_matrix[0][0], result_matrix[0][1], result_matrix[0][2], result_matrix[0][3],
             result_matrix[1][0], result_matrix[1][1], result_matrix[1][2], result_matrix[1][3],
             result_matrix[2][0], result_matrix[2][1], result_matrix[2][2], result_matrix[2][3],
             result_matrix[3][0], result_matrix[3][1], result_matrix[3][2], result_matrix[3][3]);

        // tile A1 * W3
        $display("tile A1 * tile W3");
        run_tile(W_tile3, A_tile1, 0);
        // tile A2 * W4 (use partial sums as B input)
        $display("tile A2 * tile W4");
        run_tile(W_tile4, A_tile2, 1);

        // append new partial sums into result_matrix columns 2..3
        for (int i = 0; i < 4; i++) begin
            for (int j = 0; j < 2; j++) begin
                result_matrix[i][j+2] = partial_sum[i][j];
            end
        end

        // Display final results
        $display("\n=== FINAL OUTPUT MATRIX ===");
        $display("      Col0  Col1  Col2  Col3");
        for (int i = 0; i < 4; i++) begin
            $display("Row%0d: %4d  %4d  %4d  %4d", 
                     i, result_matrix[i][0], result_matrix[i][1], 
                     result_matrix[i][2], result_matrix[i][3]);
        end

        $display("\n=== EXPECTED OUTPUT ===");
        $display("Row0:   90   100   110   120");
        $display("Row1:  202   228   254   280");
        $display("Row2:  314   356   398   440");
        $display("Row3:  426   484   542   600");

        // Verify
        $display("\n=== VERIFICATION ===");
        if (result_matrix[0][0] == 90  && result_matrix[0][1] == 100 && 
            result_matrix[0][2] == 110 && result_matrix[0][3] == 120 &&
            result_matrix[1][0] == 202 && result_matrix[1][1] == 228 && 
            result_matrix[1][2] == 254 && result_matrix[1][3] == 280 &&
            result_matrix[2][0] == 314 && result_matrix[2][1] == 356 && 
            result_matrix[2][2] == 398 && result_matrix[2][3] == 440 &&
            result_matrix[3][0] == 426 && result_matrix[3][1] == 484 && 
            result_matrix[3][2] == 542 && result_matrix[3][3] == 600) begin
            $display("✓ TEST PASSED - All outputs correct!");
        end else begin
            $display("✗ TEST FAILED - Outputs don't match expected values");
        end
        #100;
        $finish;
    end
endmodule
