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
    logic done, busy;

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
        .done(done),
        .busy(busy)
    );

    // Clock Generation
    always #5 clk = ~clk;

    // Clocking block to prevent delta-cycle race conditions
    /*clocking cb @(negedge clk);
        default input #1step output #1;
        input norm_out_valid;
        input norm_out_data;
    endclocking*/

    // --------------------------------------------------------
    // Test Data & Golden Model Variables
    // --------------------------------------------------------
    logic signed [DATAWIDTH-1:0] test_row [0:767]; 
    logic signed [DATAWIDTH-1:0] tb_gamma [0:31]; // 32 Weights
    logic signed [DATAWIDTH-1:0] tb_beta  [0:31]; // 32 Biases
    
    real expected_mean;
    real expected_var;
    real expected_inv_std;
    real expected_out [0:767];

    int out_chunk_idx = 0; 
    int errors = 0;

    int j = 0; // universal counter for stream row to compare correctly

    // --------------------------------------------------------
    // Main Test Sequence
    // --------------------------------------------------------
    initial begin
        int i;
        clk = 0;
        rst_n = 0;
        data_valid = 0;
        for (i=0; i<32; i++) buffer_rdata[i] = 0;
        
        $display("==================================================");
        $display("   STARTING CASCADED LAYERNORM VERIFICATION");
        $display("==================================================");

        // Generate Data & Math
        generate_test_data();

        // Reset
        #20 rst_n = 1;
        #10;

        // -------------------------
        // PASS 1: MEAN
        // -------------------------
        $display("\n[TIME %0t] Starting PASS 1 (Mean Computation)...", $time);
        stream_row();
        #20; // Wait for Mean broadcast

        // -------------------------
        // PASS 2: VARIANCE
        // -------------------------
        $display("[TIME %0t] Starting PASS 2 (Variance Computation)...", $time);
        stream_row();

        // -------------------------
        // PARAMETER LOADING (During SQRT)
        // -------------------------
        // Wait until FSM explicitly enters ST_CALC_SQRT (State 4)
        wait(dut.u_fsm.state == 3'd5);
        $display("[TIME %0t] FSM entered SQRT Wait. Injecting Gamma and Beta!", $time);
        
        // Cycle 1: load_parameters == 0 -> PE expects Gamma
        @(negedge clk);
        for (i = 0; i < 32; i++) buffer_rdata[i] = tb_gamma[i];
        
        
        // Cycle 2: load_parameters == 1 -> PE expects Beta
        @(negedge clk);
        for (i = 0; i < 32; i++) buffer_rdata[i] = tb_beta[i];
        
        
        // Clear bus while waiting for SQRT to finish
        @(negedge clk);
        for (i = 0; i < 32; i++) buffer_rdata[i] = '0;

        // Wait until FSM enters ST_PASS3_NORM (State 5)
        wait(dut.u_fsm.state == 3'd6);

        // -------------------------
        // PASS 3: NORMALIZATION & AFFINE
        // -------------------------
        $display("[TIME %0t] Starting PASS 3 (Normalization + Affine)...", $time);
        stream_row();

        @(negedge norm_out_valid);
        $display("\n[TIME %0t] FSM asserted DONE signal!", $time);

        $display("\n==================================================");
        if (errors == 0) $display("   SUCCESS! Hardware matches Golden Model perfectly.");
        else             $display("   FAILED! Detected %0d mismatches.", errors);
        $display("==================================================");
        $finish;
    end

    // --------------------------------------------------------
    // Task: Stream the 768-element row in 24 chunks
    // --------------------------------------------------------
    task automatic stream_row();
        int chunk, i;
        begin
            j++;
            for (chunk = 0; chunk < 24; chunk++) begin
                @(negedge clk);
                data_valid = 1'b1;
                for (i = 0; i < 32; i++) begin
                    buffer_rdata[i] = test_row[(chunk * 32) + i];
                end
                if (j == 3) begin
                    @(posedge clk);
                    compare();
                end
            end
            @(negedge clk);
            data_valid = 1'b0; 
        end
    endtask

    // --------------------------------------------------------
    // Hardware vs. Software Verification
    // --------------------------------------------------------
    task compare (); 
        begin
            int i;
            real hw_real_out;
            real diff;
            int absolute_idx;
            #1;
            
            if (norm_out_valid) begin
                for (i = 0; i < 32; i++) begin
                    absolute_idx = (out_chunk_idx * 32) + i;
                    
                    hw_real_out = real'(norm_out_data[i]) / Q_SCALE;
                    diff = hw_real_out - expected_out[absolute_idx];
                    if (diff < 0) diff = -diff;

                    // Tolerance is 0.05 due to fixed-point rounding
                    if (diff > 0.05) begin
                        $error("Mismatch at Index %0d | HW: %f | Expected: %f", absolute_idx, hw_real_out, expected_out[absolute_idx]);
                        errors++;
                    end
                    
                    if (absolute_idx < 3) begin
                        $display("   -> Index %0d | HW: %8f | Expected: %8f", absolute_idx, hw_real_out, expected_out[absolute_idx]);
                    end
                end
                out_chunk_idx++;
            end
        end
    endtask

    // --------------------------------------------------------
    // Function: Generate Data & Golden Model (Now with Affine!)
    // --------------------------------------------------------
    task automatic generate_test_data(); // <-- FIXED: Added 'automatic'
        // FIXED: All variable declarations moved strictly to the top of the scope
        int i;
        int pe_idx;
        real sum;
        real sqr_sum;
        real rand_val;
        real real_val;
        real diff;
        real real_gamma;
        real real_beta; 
        real norm_val;
        
        begin
            // Initialize variables
            sum = 0.0;
            sqr_sum = 0.0;
            
            // 1. Generate Random Inputs, Gamma (near 1.0), and Beta (near 0.1)
            for (i = 0; i < 32; i++) begin
                tb_gamma[i] = real'((1.0 + (real'($random % 100) / 1000.0)) * Q_SCALE); // ~1.0
                tb_beta[i]  = real'((0.1 + (real'($random % 100) / 1000.0)) * Q_SCALE); // ~0.1
            end

            for (i = 0; i < 768; i++) begin
                rand_val = (real'($random % 1000) / 120.0); // -8.33 to +8.33
                test_row[i] = real'(rand_val * Q_SCALE);       
                sum += rand_val;
            end

            // 2. Mean
            expected_mean = sum / 768.0;

            // 3. Variance
            for (i = 0; i < 768; i++) begin
                real_val = real'(test_row[i]) / Q_SCALE;
                diff = real_val - expected_mean;
                sqr_sum += (diff * diff);
            end
            expected_var = sqr_sum / 768.0;

            // 4. Inv_StdDev (with epsilon)
            expected_inv_std = 1.0 / $sqrt(expected_var + 0.00001);

            // 5. Expected Output (Normalization + Affine Transform)
            for (i = 0; i < 768; i++) begin
                pe_idx = i % 32; // Which PE processes this element
                
                real_val   = real'(test_row[i]) / Q_SCALE;
                real_gamma = real'(tb_gamma[pe_idx]) / Q_SCALE;
                real_beta  = real'(tb_beta[pe_idx]) / Q_SCALE;
                
                // The Complete Equation: Y = Gamma * ((X - Mean) * InvStd) + Beta
                norm_val = (real_val - expected_mean) * expected_inv_std;
                expected_out[i] = (norm_val * real_gamma) + real_beta;
            end
            
            $display("   [GOLDEN] Expected Mean     : %f", expected_mean);
            $display("   [GOLDEN] Expected Variance : %f", expected_var);
            $display("   [GOLDEN] Expected 1/StdDev : %f\n", expected_inv_std);
        end
    endtask

endmodule