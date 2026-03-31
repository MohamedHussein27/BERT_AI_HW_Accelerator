`timescale 1ns / 1ps
import PE_pkg::*;

module tb_layernorm_top;

    // --------------------------------------------------------
    // Parameters & Signals
    // --------------------------------------------------------
    parameter DATAWIDTH = 32;
    localparam real Q_SCALE = 67108864.0; // 2^26 for Q5.26 conversion

    logic clk;
    logic rst_n;
    logic data_valid;
    logic signed [DATAWIDTH-1:0] buffer_rdata [0:31];
    
    logic signed [DATAWIDTH-1:0] norm_out_data [0:31];
    logic norm_out_valid;
    logic done;

    // --------------------------------------------------------
    // Device Under Test (DUT)
    // --------------------------------------------------------
    layernorm_top #(
        .DATAWIDTH(DATAWIDTH)
    ) dut (
        .clk(clk),
        .rst_n(rst_n),
        .data_valid(data_valid),
        .buffer_rdata(buffer_rdata),
        .norm_out_data(norm_out_data),
        .norm_out_valid(norm_out_valid),
        .done(done)
    );

    // Clock Generation
    always #5 clk = ~clk;

    // --------------------------------------------------------
    // Test Data & Golden Model Variables
    // --------------------------------------------------------
    logic signed [DATAWIDTH-1:0] test_row [0:767]; // One full 768-element row
    
    real expected_mean;
    real expected_var;
    real expected_inv_std;
    real expected_out [0:767];

    int out_chunk_idx = 0; // Tracks which chunk we are verifying
    int errors = 0;

    // --------------------------------------------------------
    // Main Test Sequence
    // --------------------------------------------------------
    initial begin
        // 1. Initialize System
        clk = 0;
        rst_n = 0;
        data_valid = 0;
        for (int i=0; i<32; i++) buffer_rdata[i] = 0;
        
        $display("==================================================");
        $display("   STARTING LAYERNORM SYSTEM VERIFICATION");
        $display("==================================================");

        // 2. Generate Random Q5.26 Data & Compute Golden Math
        generate_test_data();

        // 3. Reset Hardware
        #20 rst_n = 1;
        #10;

        // 4. Execute Pass 1: Mean
        $display("\n[TIME %0t] Starting PASS 1 (Mean Computation)...", $time);
        stream_row();
        
        // FSM needs 1 cycle (ST_LOAD_MEAN) to broadcast the mean. We wait 2 just to be safe.
        #20; 

        // 5. Execute Pass 2: Variance
        $display("[TIME %0t] Starting PASS 2 (Variance Computation)...", $time);
        stream_row();

        // The Inv_Sqrt module takes multiple clock cycles to run Newton-Raphson.
        // We simulate the Main Controller waiting for 20 clock cycles.
        $display("[TIME %0t] Waiting for Newton-Raphson Sqrt to finish...", $time);
        #200; 

        // 6. Execute Pass 3: Normalization & Output Verification
        $display("[TIME %0t] Starting PASS 3 (Normalization & Output)...", $time);
        stream_row();

        // Wait for FSM to assert done
        wait(done == 1'b1);
        $display("\n[TIME %0t] FSM asserted DONE signal!", $time);

        // 7. Final Report
        $display("\n==================================================");
        if (errors == 0) begin
            $display("   SUCCESS! Hardware matches Golden Model perfectly.");
        end else begin
            $display("   FAILED! Detected %0d mismatches.", errors);
        end
        $display("==================================================");
        $finish;
    end

    // --------------------------------------------------------
    // Task: Stream the 768-element row in 24 chunks
    // --------------------------------------------------------
    task stream_row();
        begin
            for (int chunk = 0; chunk < 24; chunk++) begin
                @(negedge clk);
                data_valid = 1'b1;
                
                // Load 32 elements onto the bus
                for (int i = 0; i < 32; i++) begin
                    buffer_rdata[i] = test_row[(chunk * 32) + i];
                end
            end
            
            @(negedge clk);
            data_valid = 1'b0; // Stop streaming
        end
    endtask

    // --------------------------------------------------------
    // Hardware vs. Software Verification (Checked on Posedge)
    // --------------------------------------------------------
    always @(posedge clk) begin
        #1; // racing avoidance
        if (norm_out_valid) begin
            for (int i = 0; i < 32; i++) begin
                int absolute_idx = (out_chunk_idx * 32) + i;
                
                // Convert hardware fixed-point output to floating-point
                real hw_real_out = real'(norm_out_data[i]) / Q_SCALE;
                
                // Calculate absolute difference
                real diff = hw_real_out - expected_out[absolute_idx];
                if (diff < 0) diff = -diff;

                // Tolerance is 0.05 due to fixed-point rounding and Newton-Raphson approximations
                if (diff > 0.05) begin
                    $error("Mismatch at Index %0d | HW: %f | Expected: %f", absolute_idx, hw_real_out, expected_out[absolute_idx]);
                    errors++;
                end
                
                // Print the very first 3 elements so we can visually confirm it works in the transcript
                if (absolute_idx < 3) begin
                    $display("   -> Index %0d | HW Output: %8f | Golden Expected: %8f", absolute_idx, hw_real_out, expected_out[absolute_idx]);
                end
            end
            out_chunk_idx++;
        end
    end

    // --------------------------------------------------------
    // Function: Generate Data & Floating-Point Golden Model
    // --------------------------------------------------------
    function void generate_test_data();
        real sum = 0.0;
        real sqr_sum = 0.0;
        
        // 1. Generate random inputs between roughly -2.0 and +2.0
        for (int i = 0; i < 768; i++) begin
            real rand_val = (real'($random % 1000) / 500.0); // -2.0 to 2.0
            test_row[i] = signed'(rand_val * Q_SCALE);       // Convert to Q5.26
            
            sum += rand_val;
        end

        // 2. Compute Floating-Point Mean
        expected_mean = sum / 768.0;

        // 3. Compute Floating-Point Variance
        for (int i = 0; i < 768; i++) begin
            real real_val = real'(test_row[i]) / Q_SCALE;
            real diff = real_val - expected_mean;
            sqr_sum += (diff * diff);
        end
        expected_var = sqr_sum / 768.0;

        // 4. Compute Floating-Point Inverse Standard Deviation (with epsilon)
        expected_inv_std = 1.0 / $sqrt(expected_var + 0.00001);

        // 5. Compute the final Expected Normalized Outputs
        for (int i = 0; i < 768; i++) begin
            real real_val = real'(test_row[i]) / Q_SCALE;
            expected_out[i] = (real_val - expected_mean) * expected_inv_std;
        end
        
        $display("   [GOLDEN MODEL] Expected Mean        : %f", expected_mean);
        $display("   [GOLDEN MODEL] Expected Variance    : %f", expected_var);
        $display("   [GOLDEN MODEL] Expected 1/StdDev    : %f\n", expected_inv_std);
    endfunction

endmodule