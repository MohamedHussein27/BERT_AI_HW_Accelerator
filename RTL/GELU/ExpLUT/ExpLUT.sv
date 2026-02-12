`timescale 1ns/1ps
// Shared LUT Module - Single instance for all 32 EU modules
module SharedLUT #(
  parameter int Q = 22,
  parameter int W = 32,
  parameter int NUM_SEGMENTS = 8,
  parameter int NUM_PORTS = 64  // Number of parallel access ports
) (
  input  wire [2:0]               segment_index [NUM_PORTS-1:0],  // Changed logic to wire
  output logic signed [W-1:0]     k_coeff [NUM_PORTS-1:0],
  output logic signed [W-1:0]     b_intercept [NUM_PORTS-1:0]
);

  // LUT for 8 segments
localparam logic signed [W-1:0] k_fixed [0:7] = '{
    32'h002E5707, // 0.724062
    32'h003288B9, // 0.789595
    32'h00371B99, // 0.861059
    32'h003C1872, // 0.938992
    32'h004188DB, // 1.023978
    32'h0047774A, // 1.116656
    32'h004DEF28, // 1.217722
    32'h0054FCE4 // 1.327935
};

localparam logic signed [W-1:0] b_fixed [0:7] = '{
    32'h00400000, // 1.000000
    32'h003F79C9, // 0.991808
    32'h003E5511, // 0.973942
    32'h003C7640, // 0.944717
    32'h0039BE0B, // 0.902224
    32'h00360906, // 0.844301
    32'h00312F20, // 0.768501
    32'h002B031B // 0.672065
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
