`timescale 1ns / 1ps
import PE_pkg::*;

module tb_layernorm_real;

    parameter DATAWIDTH = 32;
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
    // Memory Arrays for File Data
    // Sized to 768 to accommodate a full BERT Base row
    // --------------------------------------------------------
    logic signed [DATAWIDTH-1:0] test_row         [0:767]; 
    logic signed [DATAWIDTH-1:0] tb_gamma         [0:767]; 
    logic signed [DATAWIDTH-1:0] tb_beta          [0:767]; 
    logic signed [DATAWIDTH-1:0] expected_out_hex [0:767];

    int out_chunk_idx = 0; 
    int errors = 0;

    initial begin
        int i;
        clk = 0;
        rst_n = 0;
        data_valid = 0;
        for (i=0; i<32; i++) buffer_rdata[i] = 0;
        
        $display("==================================================");
        $display("   STARTING FILE-DRIVEN LAYERNORM VERIFICATION");
        $display("==================================================");

        // Load data from .hex files
        load_test_data();

        // Reset
        #20 rst_n = 1;
        #10;

        $display("\n[TIME %0t] Starting PASS 1 (Mean Computation)...", $time);
        stream_row();
        #20; 

        $display("[TIME %0t] Starting PASS 2 (Variance Computation)...", $time);
        stream_row();

        wait(dut.u_fsm.state == 3'd4); // ST_CALC_SQRT
        $display("[TIME %0t] FSM entered SQRT Wait. Injecting Gamma and Beta!", $time);
        
        @(negedge clk);
        data_valid = 1'b1; 
        // Note: Feeding the first 32 parameters to match your FSM's 1-cycle load
        for (i = 0; i < 32; i++) buffer_rdata[i] = tb_gamma[i];
        
        @(negedge clk);
        for (i = 0; i < 32; i++) buffer_rdata[i] = tb_beta[i];
        
        @(negedge clk);
        data_valid = 1'b0;
        for (i = 0; i < 32; i++) buffer_rdata[i] = '0;

        wait(dut.u_fsm.state == 3'd5); // ST_PASS3_NORM

        $display("[TIME %0t] Starting PASS 3 (Normalization + Affine)...", $time);
        stream_row();

        // Wait for background verification block to verify all chunks
        wait(out_chunk_idx == 23);
        $display("\n[TIME %0t] Hardware finished processing all data!", $time);

        $display("\n==================================================");
        if (errors == 0) $display("   SUCCESS! Hardware matches Python Golden Model perfectly.");
        else             $display("   FAILED! Detected %0d mismatches.", errors);
        $display("==================================================");
        #50;
        $finish;
    end

    // --------------------------------------------------------
    // Task: Stream the 768-element row in 24 chunks
    // --------------------------------------------------------
    task automatic stream_row();
        int chunk, i;
        begin
            for (chunk = 0; chunk < 24; chunk++) begin
                @(negedge clk);
                data_valid = 1'b1;
                for (i = 0; i < 32; i++) begin
                    buffer_rdata[i] = test_row[(chunk * 32) + i];
                end
            end
            @(negedge clk);
            data_valid = 1'b0; 
        end
    endtask

    // --------------------------------------------------------
    // Background Verification Block
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
                absolute_idx = (out_chunk_idx * 32) + i;
                
                // Convert both hardware output and Python expected hex back to real numbers for comparison
                hw_real_out       = real'(norm_out_data[i]) / Q_SCALE;
                expected_real_out = real'(expected_out_hex[absolute_idx]) / Q_SCALE;
                
                diff = hw_real_out - expected_real_out;
                if (diff < 0) diff = -diff;

                // Tolerance is 0.05 to account for slight rounding differences between Python and Verilog
                if (diff > 0.05) begin
                    $error("Mismatch at Index %0d | HW: %f | Python Exp: %f", absolute_idx, hw_real_out, expected_real_out);
                    errors++;
                end
                
                if (absolute_idx < 3) begin
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
            // Ensure the .hex files are in the same directory as the simulation executable,
            // or provide the full absolute path here (e.g., "D:/.../inputs.hex")
            $readmemh("inputs.hex", test_row);
            $readmemh("real_gamma.hex", tb_gamma);
            $readmemh("real_beta.hex", tb_beta);
            $readmemh("expected.hex", expected_out_hex);
            
            $display("   [FILE I/O] Successfully loaded hex data from Python.");
            
        end
    endtask

endmodule