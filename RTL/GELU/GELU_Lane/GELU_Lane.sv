`timescale 1ns/1ps
module GELU #(
    parameter int Q = 16,              // Q48.16 format
    parameter int W = 64,              // 64-bit width
    parameter int LUT_PORT_BASE = 0    // Base index for this GELU's LUT ports
) (
    // Data path
    input  wire signed [W-1:0] x,              // Input value in Q48.16
    output wire signed [W-1:0] y,              // Output from final EU
    
    // LUT interface - segment indices (outputs to SharedLUT)
    output wire [2:0] segment_index_0,         // EU1 segment index
    output wire [2:0] segment_index_1,         // EU2 segment index
    
    // LUT interface - coefficients (inputs from SharedLUT)
    input  wire signed [W-1:0] k_coeff_0,      // EU1 K coefficient (64-bit from testbench)
    input  wire signed [W-1:0] b_intercept_0,  // EU1 B intercept (64-bit from testbench)
    input  wire signed [W-1:0] k_coeff_1,      // EU2 K coefficient (64-bit from testbench)
    input  wire signed [W-1:0] b_intercept_1   // EU2 B intercept (64-bit from testbench)
);

    // STAGE 1: Polynomial Unit - Compute s_x = f(x)
    // Output: Q48.16 (64-bit)
    wire signed [W-1:0] s_x_q4816;  // 64-bit Q48.16
    
    PolynomialUnit #(
        .WIDTH(W),
        .Q(Q)
    ) poly_inst (
        .x(x),
        .s_x(s_x_q4816)
    );

    // Format Conversion: Q48.16 (64-bit) → Q10.22 (32-bit) for EU1
    // Shift right by (16 - 22) = -6, so LEFT shift by 6
    wire signed [31:0] s_x_q1022;
    assign s_x_q1022 = s_x_q4816[31:0] << 6;  // Convert Q48.16 → Q10.22

    // STAGE 2: First Exponential Unit - Compute exp(s_x)
    // Input: Q10.22 (32-bit), Output: Q48.16 (64-bit)
    wire signed [2*32-1:0] exp_s_x;  // 64-bit Q48.16
    
    EU #(
        .WIDTH(32),
        .Q_IN(22),
        .Q_OUT(16)
    ) eu1_inst (
        .s_x(s_x_q1022),
        .segment_index(segment_index_0),
        .K(k_coeff_0[31:0]),
        .B(b_intercept_0[31:0]),
        .exp_result(exp_s_x)
    );

    // STAGE 3: Division Unit - Compute x / (1 + exp(s_x))
    // Input: Q48.16 (64-bit), Output: Q48.16 (64-bit)
    wire signed [W-1:0] du_exponent;
    wire du_sign;
    
    DU #(
        .Q(Q),
        .W(W)
    ) du_inst (
        .F(x),
        .s_xi(exp_s_x),
        .exponent(du_exponent),
        .result_sign(du_sign)
    );

    // Format Conversion: Q48.16 (64-bit) → Q10.22 (32-bit) for EU2
    wire signed [31:0] du_exp_q1022;
    assign du_exp_q1022 = du_exponent[31:0] << 6;  // Convert Q48.16 → Q10.22

    // STAGE 4: Second Exponential Unit - Antilog (2^exponent)
    // Input: Q10.22 (32-bit), Output: Q48.16 (64-bit)
    wire signed [2*32-1:0] eu2_result;  // 64-bit Q48.16
    
    EU #(
        .WIDTH(32),
        .Q_IN(22),
        .Q_OUT(16)
    ) eu2_inst (
        .s_x(du_exp_q1022),
        .segment_index(segment_index_1),
        .K(k_coeff_1[31:0]),
        .B(b_intercept_1[31:0]),
        .exp_result(eu2_result)
    );

    // OUTPUT: Apply sign from DU
    // GELU(x) = x / (1 + exp(s_x)) with proper sign handling
    assign y = du_sign ? -eu2_result : eu2_result;

endmodule
