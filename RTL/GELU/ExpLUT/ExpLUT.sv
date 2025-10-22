`timescale 1ns/1ps
// ============================================================================
// Shared LUT Module - Single instance for all 32 EU modules
// ============================================================================
module ExpLUT #(
  parameter int Q = 26,
  parameter int W = 32,
  parameter int NUM_SEGMENTS = 8,
  parameter int NUM_PORTS = 32  // Number of parallel access ports
) (
  input  wire [2:0]               segment_index [NUM_PORTS-1:0],  // Changed logic to wire
  output logic signed [W-1:0]     k_coeff [NUM_PORTS-1:0],
  output logic signed [W-1:0]     b_intercept [NUM_PORTS-1:0]
);

  // LUT for 8 segments
  localparam logic signed [W-1:0] k_fixed [0:7] = '{
      32'h02E57078, // 0.724062
      32'h03288B9B, // 0.789595
      32'h0371B996, // 0.861060
      32'h03C18722, // 0.938992
      32'h04188DB7, // 1.023978
      32'h047774AE, // 1.116656
      32'h04DEF287, // 1.217722
      32'h054FCE46  // 1.327935
  };

  localparam logic signed [W-1:0] b_fixed [0:7] = '{
      32'h04000000, // 1.000000
      32'h03F79C9B, // 0.991808
      32'h03E5511D, // 0.973942
      32'h03C76408, // 0.944718
      32'h039BE0BD, // 0.902224
      32'h03609063, // 0.844301
      32'h0312F200, // 0.768501
      32'h02B031B9  // 0.672065
  };

  // Combinational multi-port lookup
  genvar i;
  generate
    for (i = 0; i < NUM_PORTS; i++) begin : lut_ports
      assign k_coeff[i] = k_fixed[segment_index[i]];
      assign b_intercept[i] = b_fixed[segment_index[i]];
    end
  endgenerate

endmodule
