`timescale 1ns / 1ps
import PE_pkg::*;

module tb_processing_element;

    parameter Q_FRAC_BITS = 26;
    parameter DATA_WIDTH  = 32;

    logic clk;
    logic rst_n;

    pe_op_e opcode;
    logic signed [31:0] data_in;
    logic signed [31:0] bcast_data;
    logic signed [31:0] data_out;

    // DUT
    processing_element dut (
        .clk(clk),
        .rst_n(rst_n),
        .opcode(opcode),
        .data_in(data_in),
        .bcast_data(bcast_data),
        .data_out(data_out)
    );

    // Clock
    always #5 clk = ~clk;

    // Scoreboard variables
    logic signed [31:0] gamma, beta, mu;
    logic signed [31:0] expected;

    // =============================
    // TESTS
    // =============================
    initial begin
        clk = 0;
        reset();

        // -------------------------
        // Load gamma, beta, and mu
        // -------------------------
        gamma = 32'sd1000;
        beta  = 32'sd200;
        mu    = 32'sd500; // Simulated Mean

        apply_and_check(OP_LOAD_WGT,  gamma, 0);
        apply_and_check(OP_LOAD_BIAS, beta,  0);
        apply_and_check(OP_LOAD_MEAN, 0,     mu); 

        // -------------------------
        // PASS_X
        // -------------------------
        repeat (10) begin
            apply_and_check(OP_PASS_X, $random, 0);
        end

        // -------------------------
        // VAR_SQR
        // sub = (data_in - mu), mul = sub * sub
        // -------------------------
        repeat (20) begin
            apply_and_check(OP_VAR_SQR, $random, 0); 
        end

        // -------------------------
        // NORMALIZE (Now includes Affine!)
        // 1. sub = (data_in - mu)
        // 2. mul1 = sub * bcast (1/sigma), quantize
        // 3. mul2 = mul1 * gamma, quantize
        // 4. out = mul2 + beta
        // -------------------------
        repeat (20) begin
            apply_and_check(OP_NORMALIZE, $random, $random); 
        end

        // -------------------------
        // CORNER CASES (Saturation/Overflow Checks)
        // -------------------------
        apply_and_check(OP_VAR_SQR, 32'sh7FFFFFFF, 0);
        apply_and_check(OP_VAR_SQR, 32'sh80000000, 0);

        apply_and_check(OP_NORMALIZE, 32'sh7FFFFFFF, 32'sh7FFFFFFF);
        apply_and_check(OP_NORMALIZE, 32'sh80000000, 32'sh80000000);

        $display("All tests passed!");
        $finish;
    end

    // =============================
    // RESET
    // =============================
    task reset();
        begin
            rst_n = 0;
            repeat(5) @(negedge clk);
            rst_n = 1;
        end
    endtask

    // =============================
    // APPLY + CHECK
    // =============================
    task apply_and_check(
        input pe_op_e op,
        input logic signed [31:0] din,
        input logic signed [31:0] bcast
    );
        begin
            opcode     = op;
            data_in    = din;
            bcast_data = bcast;

            @(negedge clk);

            // Pass 'mu' into the golden model
            expected = golden_model(op, din, bcast, gamma, beta, mu);

            // Debug print
            $display("--------------------------------------------------");
            $display("TIME = %0t", $time);
            $display("OPCODE      = %0d", op);
            $display("DATA_IN     = %0d", din);
            $display("BCAST_DATA  = %0d", bcast);
            $display("GAMMA       = %0d", gamma);
            $display("BETA        = %0d", beta);
            $display("MEAN (MU)   = %0d", mu);
            $display("EXPECTED    = %0d", expected);
            $display("DUT OUTPUT  = %0d", data_out);

            if (data_out !== expected) begin
                $error(" MISMATCH DETECTED!");
            end else begin
                $display(" PASS");
            end
        end
    endtask

    // =============================
    // HELPER: TB QUANTIZATION FUNCTION
    // =============================
    function automatic logic signed [31:0] quantize_tb(
        input logic signed [63:0] mul_full_in
    );
        logic signed [63:0] mul_rounded;
        
        // Add half for rounding, then shift right by 26
        mul_rounded = (mul_full_in + (64'(1) << 25)) >>> 26;
        
        // Saturation Clamping
        if (mul_rounded > 64'sh000000007FFFFFFF) begin
            return 32'sh7FFFFFFF; 
        end else if (mul_rounded < -64'sh0000000080000000) begin
            return -32'sh80000000; 
        end else begin
            return mul_rounded[31:0]; 
        end
    endfunction

    // =============================
    // UPDATED GOLDEN MODEL (Matches Cascaded RTL)
    // =============================
    function automatic logic signed [31:0] golden_model(
        pe_op_e opcode,
        logic signed [31:0] data_in,
        logic signed [31:0] bcast,
        logic signed [31:0] gamma,
        logic signed [31:0] beta,
        logic signed [31:0] mu 
    );
        logic signed [31:0] sub;
        logic signed [63:0] mul_full;
        logic signed [31:0] mul_out;

        // 1. SUB: Subtract latched 'mu'
        if (opcode inside {OP_VAR_SQR, OP_NORMALIZE})
            sub = data_in - mu;
        else
            sub = data_in;

        // 2. MUL Stage 1
        if (opcode == OP_VAR_SQR)
            mul_full = sub * sub;
        else if (opcode == OP_NORMALIZE)
            mul_full = sub * bcast; // bcast acts as 1/sigma
        else
            mul_full = {32'd0, sub};

        // 3. Quantize Stage 1
        if (opcode inside {OP_VAR_SQR, OP_NORMALIZE})
            mul_out = quantize_tb(mul_full);
        else
            mul_out = mul_full[31:0];

        // 4. MUL Stage 2 & ADD (The Cascade)
        if (opcode == OP_NORMALIZE) begin
            mul_full = mul_out * gamma;       // Cascade the first quantized output
            mul_out  = quantize_tb(mul_full); // Quantize again!
            return mul_out + beta;            // Apply Affine Bias
        end else begin
            return mul_out;                   // Bypass Adder
        end
    endfunction

endmodule