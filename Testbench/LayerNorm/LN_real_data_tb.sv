`timescale 1ns / 1ps
import PE_pkg::*;

module tb_layernorm_real;

    // --------------------------------------------------------
    // Parameters & Signals
    // --------------------------------------------------------
    parameter DATAWIDTH = 32;
    parameter NUM_ROWS  = 10; // Change this to run 2, 10, or 100 rows!
    
    localparam TOTAL_ELEMENTS = NUM_ROWS * 768;
    localparam real Q_SCALE = 67108864.0; // 2^26 

    logic clk;
    logic rst_n;
    logic data_valid;
    logic signed [DATAWIDTH-1:0] buffer_rdata [0:31];
    logic signed [DATAWIDTH-1:0] norm_out_data [0:31];
    logic norm_out_valid;
    logic done, busy;

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

    always #5 clk = ~clk;

    // --------------------------------------------------------
    // Memory Arrays for File Data (Scaled for NUM_ROWS)
    // --------------------------------------------------------
    logic signed [DATAWIDTH-1:0] test_data        [0:TOTAL_ELEMENTS-1]; 
    logic signed [DATAWIDTH-1:0] expected_out_hex [0:TOTAL_ELEMENTS-1];
    
    // Parameters stay at 768 (or 32 depending on your architecture)
    logic signed [DATAWIDTH-1:0] tb_gamma         [0:767]; 
    logic signed [DATAWIDTH-1:0] tb_beta          [0:767]; 

    int out_chunk_idx = 0; 
    int errors = 0;

    initial begin
        int i, r;
        clk = 0;
        rst_n = 0;
        data_valid = 0;
        for (i=0; i<32; i++) buffer_rdata[i] = 0;
        
        $display("==================================================");
        $display("   STARTING MULTI-ROW FILE I/O VERIFICATION");
        $display("   TESTING %0d ROWS", NUM_ROWS);
        $display("==================================================");

        // Load data from .hex files
        load_test_data();

        // Reset
        #20 rst_n = 1;
        #10;

        // --------------------------------------------------------
        // THE MULTI-ROW LOOP
        // --------------------------------------------------------
        for (r = 0; r < NUM_ROWS; r++) begin
            $display("\n==================================================");
            $display("   PROCESSING ROW %0d", r);
            $display("==================================================");

            // calculate mean and variance for each row
            calculate_and_print_golden_stats(r);

            $display("[TIME %0t] Starting PASS 1 (Mean Computation)...", $time);
            stream_row(r); // Pass the row index!
            #20; 

            $display("[TIME %0t] Starting PASS 2 (Variance Computation)...", $time);
            stream_row(r);

            wait(dut.u_fsm.state == 3'd4); // ST_CALC_SQRT
            $display("[TIME %0t] FSM entered SQRT Wait. Injecting Gamma and Beta!", $time);
            
            @(negedge clk);
            data_valid = 1'b1; 
            for (i = 0; i < 32; i++) buffer_rdata[i] = tb_gamma[i];
            
            @(negedge clk);
            for (i = 0; i < 32; i++) buffer_rdata[i] = tb_beta[i];
            
            @(negedge clk);
            data_valid = 1'b0;
            for (i = 0; i < 32; i++) buffer_rdata[i] = '0;

            wait(dut.u_fsm.state == 3'd5); // ST_PASS3_NORM

            $display("[TIME %0t] Starting PASS 3 (Normalization + Affine)...", $time);
            stream_row(r);

            //#50;

            // Wait for FSM to cycle back to IDLE/MEAN before starting the next row
            wait(dut.u_fsm.state == 3'd0 || done == 1'b1);
            out_chunk_idx++;
        end

        // Wait for background verification block to verify all chunks across all rows
        wait(out_chunk_idx == ((NUM_ROWS * 24) /* - (1 * NUM_ROWS)*/));
        $display("\n[TIME %0t] Hardware finished processing all %0d rows!", $time, NUM_ROWS);

        $display("\n==================================================");
        if (errors == 0) $display("   SUCCESS! Hardware matches Python Golden Model perfectly.");
        else             $display("   FAILED! Detected %0d mismatches.", errors);
        $display("==================================================");
        #50;
        $finish;
    end

    // --------------------------------------------------------
    // Task: Stream the 768-element row in 24 chunks
    // NOW ACCEPTS A ROW INDEX to calculate the memory offset
    // --------------------------------------------------------
    task automatic stream_row(int row_idx);
        int chunk, i;
        int base_idx;
        begin
            base_idx = row_idx * 768; // Calculate where this row starts in memory
            
            for (chunk = 0; chunk < 24; chunk++) begin
                @(negedge clk);
                data_valid = 1'b1;
                for (i = 0; i < 32; i++) begin
                    // Read from the offset base_idx
                    buffer_rdata[i] = test_data[base_idx + (chunk * 32) + i];
                end
            end
            @(negedge clk);
            data_valid = 1'b0; 
        end
    endtask

    // --------------------------------------------------------
    // Background Verification Block (Unaffected by multiple rows!)
    // --------------------------------------------------------
    always @(negedge clk) begin
        if (rst_n && norm_out_valid) begin
            compare();
        end
    end

    // --------------------------------------------------------
    // Task: Compare Hardware Output vs Python Expected Hex
    // --------------------------------------------------------
    task compare(); 
        begin
            int i;
            real hw_real_out;
            real expected_real_out;
            real diff;
            int absolute_idx;
            
            for (i = 0; i < 32; i++) begin
                // out_chunk_idx naturally counts up from 0 to (NUM_ROWS * 24)
                absolute_idx = (out_chunk_idx * 32) + i;
                
                hw_real_out       = real'(norm_out_data[i]) / Q_SCALE;
                expected_real_out = real'(expected_out_hex[absolute_idx]) / Q_SCALE;
                
                diff = hw_real_out - expected_real_out;
                if (diff < 0) diff = -diff;

                if (diff > 0.05) begin
                    $error("Mismatch at Index %0d | HW: %f | Python Exp: %f | index: %d", absolute_idx, hw_real_out, expected_real_out, out_chunk_idx);
                    errors++;
                end
                
                // Only print the first 3 elements of EACH ROW to keep the transcript clean
                if ((absolute_idx % 768) < 3) begin
                    $display("   -> Index %0d | HW: %8f | Python Exp: %8f", absolute_idx, hw_real_out, expected_real_out);
                end
            end
            out_chunk_idx++;
        end
    endtask

    // --------------------------------------------------------
    // Task: Load Hex Files
    // --------------------------------------------------------
    task automatic load_test_data(); 
        begin
            $readmemh("inputs.hex", test_data);
            $readmemh("real_gamma.hex", tb_gamma);
            $readmemh("real_beta.hex", tb_beta);
            $readmemh("expected.hex", expected_out_hex);
            
            $display("   [FILE I/O] Successfully loaded hex data from Python.");
        end
    endtask

    // --------------------------------------------------------
    // Task: Calculate and Print Golden Stats for a Row
    // --------------------------------------------------------
    task automatic calculate_and_print_golden_stats(int row_idx);
        int i;
        int base_idx;
        real sum = 0.0;
        real sqr_sum = 0.0;
        real real_val, diff;
        real expected_mean, expected_var, expected_inv_std;

        begin
            base_idx = row_idx * 768;

            // 1. Mean
            for (i = 0; i < 768; i++) begin
                real_val = real'(test_data[base_idx + i]) / Q_SCALE;
                sum += real_val;
            end
            expected_mean = sum / 768.0;

            // 2. Variance
            for (i = 0; i < 768; i++) begin
                real_val = real'(test_data[base_idx + i]) / Q_SCALE;
                diff = real_val - expected_mean;
                sqr_sum += (diff * diff);
            end
            expected_var = sqr_sum / 768.0;

            // 3. Inv_StdDev (with epsilon)
            expected_inv_std = 1.0 / $sqrt(expected_var + 0.00001);

            $display("   [MATH GOLDEN] Expected Mean     : %f", expected_mean);
            $display("   [MATH GOLDEN] Expected Variance : %f", expected_var);
            $display("   [MATH GOLDEN] Expected 1/StdDev : %f", expected_inv_std);
        end
    endtask

endmodule