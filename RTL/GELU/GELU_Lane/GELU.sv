`timescale 1ns/1ps

module GELU #(
    parameter int Q = 22,              // Q10.22 format for I/O
    parameter int W = 32,              // 32-bit I/O width
    parameter int LUT_PORT_BASE = 0
) (
    // Data path - 32-bit I/O
    input  wire signed [W-1:0] x,              // Input Q10.22
    output wire signed [2*W-1:0] y,              // Output Q48.16
    
    // LUT interface
    output wire [2:0] segment_index_0,
    output wire [2:0] segment_index_1,
    
    input  wire signed [W-1:0] k_coeff_0,
    input  wire signed [W-1:0] b_intercept_0,
    input  wire signed [W-1:0] k_coeff_1,
    input  wire signed [W-1:0] b_intercept_1
);

    // STAGE 1: Polynomial Unit - Q10.22 → Q10.22
    wire signed [W-1:0] s_x_q1022;  // 32-bit Q10.22
    
    PolynomialUnit #(
        .WIDTH(32),
        .Q(22)
    ) poly_inst (
        .x(x),
        .s_x(s_x_q1022)
    );

    // STAGE 2: EU1 - Q10.22 → Q48.16
    wire signed [2*W-1:0] exp_s_x;  // 64-bit Q48.16
    
    EU #(
        .WIDTH(32),
        .Q_IN(22),
        .Q_OUT(16)
    ) eu1_inst (
        .s_x(s_x_q1022),
        .segment_index(segment_index_0),
        .K(k_coeff_0),
        .B(b_intercept_0),
        .exp_result(exp_s_x)
    );

    // Format Conversion: Q10.22 (32-bit) → Q48.16 (64-bit) for DU
    wire signed [2*W-1:0] x_q4816;  // 64-bit Q48.16
    
    // Convert: Q10.22 → Q48.16 (shift right by 6)
    assign x_q4816 = (64'(signed'(x))) >>> 6;

    // STAGE 3: DU - Q48.16 → Q48.16
    wire signed [2*W-1:0] du_exponent;  // 64-bit Q48.16
    wire du_sign;
    
    DU #(
        .Q(16),
        .W(64)
    ) du_inst (
        .F(x_q4816),           // ✅ 64-bit Q48.16 input
        .s_xi(exp_s_x),        // ✅ 64-bit Q48.16 input
        .exponent(du_exponent),
        .result_sign(du_sign)
    );

    // Format Conversion: Q48.16 (64-bit) → Q10.22 (32-bit) for EU2
    wire signed [W-1:0] du_exp_q1022;
    
    // Convert: Q48.16 → Q10.22 (shift left by 6)
    assign du_exp_q1022 = du_exponent[31:0] << 6;

    // STAGE 4: EU2 - Q10.22 → Q48.16
    wire signed [2*W-1:0] eu2_result_64;  // 64-bit Q48.16
    
    EU #(
        .WIDTH(32),
        .Q_IN(22),
        .Q_OUT(16)
    ) eu2_inst (
        .s_x(du_exp_q1022),
        .segment_index(segment_index_1),
        .K(k_coeff_1),
        .B(b_intercept_1),
        .exp_result(eu2_result_64)
    );


    // OUTPUT: Apply sign from DU
    assign y = du_sign ? -eu2_result_64 : eu2_result_64;

endmodule
