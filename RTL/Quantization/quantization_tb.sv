`timescale 1ns / 1ps

module tb_quantize();

    // -----------------------------------------
    // Parameters (Matching the DUT)
    // -----------------------------------------
    localparam DATAWIDTH_in = 32;
    localparam DATAWIDTH_out = 8;
    localparam M_width = 8;
    localparam S_width = 8;

    // -----------------------------------------
    // Signals
    // -----------------------------------------
    logic clk;
    logic rst_n;
    logic valid_in;
    logic signed [DATAWIDTH_in-1:0] data_in;
    logic [M_width-1:0] scale_M;
    logic [S_width-1:0] scale_S;
    
    logic signed [DATAWIDTH_out-1:0] data_out;
    logic valid_out;

    // -----------------------------------------
    // DUT Instantiation
    // -----------------------------------------
    quantize #(
        .DATAWIDTH_in(DATAWIDTH_in),
        .DATAWIDTH_out(DATAWIDTH_out),
        .M_width(M_width),
        .S_width(S_width)
    ) dut (
        .clk(clk),
        .rst_n(rst_n),
        .valid_in(valid_in),
        .data_in(data_in),
        .scale_M(scale_M),
        .scale_S(scale_S),
        .data_out(data_out),
        .valid_out(valid_out)
    );

    // -----------------------------------------
    // Clock Generation (100 MHz)
    // -----------------------------------------
    initial begin
        clk = 0;
        forever #5 clk = ~clk; 
    end

    // -----------------------------------------
    // Main Test Sequence
    // -----------------------------------------
    initial begin
        // 1. Initialize all signals
        rst_n    = 0;
        valid_in = 0;
        data_in  = 0;
        scale_M  = 0;
        scale_S  = 0;

        // 2. Apply Reset
        #20;
        rst_n = 1;
        @(negedge clk);

        $display("Starting Quantization Module Tests...\n");

        // --- TEST CASE 1: Standard Positive Value ---
        // data = 100,000,000. M = 128. S = 0.
        // Expected math: (100,000,000 * 128) / 2^31 ≈ 5.96 (Rounds to 6)
        apply_stimulus(100_000_000, 128, 0);
        @(posedge valid_out); 
        $display("Time: %0t | TC1 (Pos)   | In: %0d | M: %0d | S: %0d | Out: %0d", 
                 $time, 100_000_000, 128, 0, data_out);


        // --- TEST CASE 2: Standard Negative Value ---
        // Expected math: (-100,000,000 * 128) / 2^31 ≈ -5.96 (Rounds to -6)
        @(negedge clk);
        apply_stimulus(-100_000_000, 128, 0);
        @(posedge valid_out);
        $display("Time: %0t | TC2 (Neg)   | In: %0d | M: %0d | S: %0d | Out: %0d", 
                 $time, -100_000_000, 128, 0, data_out);


        // --- TEST CASE 3: Positive Clamping (Saturation) ---
        // data = 2,000,000,000. M = 200. S = 0.
        // Expected math: (2B * 200) / 2^31 ≈ 186. Should clamp to 127.
        @(negedge clk);
        apply_stimulus(2_000_000_000, 200, 0);
        @(posedge valid_out);
        $display("Time: %0t | TC3 (Clamp +) | In: %0d | M: %0d | S: %0d | Out: %0d", 
                 $time, 2_000_000_000, 200, 0, data_out);


        // --- TEST CASE 4: Negative Clamping (Saturation) ---
        // Expected math: (-2B * 200) / 2^31 ≈ -186. Should clamp to -128.
        @(negedge clk);
        apply_stimulus(-2_000_000_000, 200, 0);
        @(posedge valid_out);
        $display("Time: %0t | TC4 (Clamp -) | In: %0d | M: %0d | S: %0d | Out: %0d", 
                 $time, -2_000_000_000, 200, 0, data_out);

        // End simulation
        #50;
        $display("\nTests completed.");
        $finish;
    end


    // -----------------------------------------
    // Task: Apply Stimulus
    // -----------------------------------------
    // Injects data, pulses valid_in for 1 clock cycle
    task apply_stimulus(
        input logic signed [DATAWIDTH_in-1:0] in_val,
        input logic [M_width-1:0] m_val,
        input logic [S_width-1:0] s_val
    );
        begin
            valid_in <= 1'b1;
            data_in  <= in_val;
            scale_M  <= m_val;
            scale_S  <= s_val;
            
            @(negedge clk);
            valid_in <= 1'b0; // De-assert valid after 1 cycle
        end
    endtask

endmodule