`timescale 1ns/1ps
// ============================================================================
// Top-level: 32 Parallel GELU Lanes with Shared LUT
// ============================================================================
module GELU_Array #(
  parameter int Q = 26,
  parameter int W = 32,
  parameter int INT_WIDTH = 5,
  parameter int NUM_LANES = 32
) (
  input  logic                            clk,
  input  logic                            rst_n,
  input  logic                            valid_in,
  input  wire signed [W-1:0]             xi [NUM_LANES-1:0],  // 32 inputs
  
  output logic                            valid_out,
  output logic signed [W-1:0]             gelu_out [NUM_LANES-1:0]  // 32 outputs
);

  // Shared LUT signals
  logic [2:0]           segment_index [NUM_LANES-1:0];
  logic signed [W-1:0]  k_coeff [NUM_LANES-1:0];
  logic signed [W-1:0]  b_intercept [NUM_LANES-1:0];
  
  // Valid signals from each lane
  logic [NUM_LANES-1:0] valid_lane;

  // Single shared LUT instance (accessed by all 32 lanes)
  ExpLUT #(
    .Q(Q),
    .W(W),
    .NUM_SEGMENTS(8),
    .NUM_PORTS(NUM_LANES)
  ) shared_lut (
    .segment_index(segment_index),
    .k_coeff(k_coeff),
    .b_intercept(b_intercept)
  );

  // Generate 32 parallel GELU lanes
  genvar i;
  generate
    for (i = 0; i < NUM_LANES; i++) begin : gelu_lanes
      GELU_Lane #(
        .Q(Q),
        .W(W),
        .INT_WIDTH(INT_WIDTH)
      ) lane (
        .clk(clk),
        .rst_n(rst_n),
        .valid_in(valid_in),
        .xi(xi[i]),
        .segment_index(segment_index[i]),
        .k_coeff(k_coeff[i]),
        .b_intercept(b_intercept[i]),
        .valid_out(valid_lane[i]),
        .gelu_result(gelu_out[i])
      );
    end
  endgenerate

  // Output valid when all lanes are valid
  assign valid_out = &valid_lane;  // AND of all valid signals

endmodule
