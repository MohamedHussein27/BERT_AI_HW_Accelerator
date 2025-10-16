`timescale 1ns/1ps

module PolynomialUnit #(
  parameter int Q = 26,      // fractional bits (Q5.26)
  parameter int W = 32       // data width (bits)
) (
  input  logic                 clk,
  input  logic                 rst_n,     // async active-low reset
  input  logic                 valid_in,  // input valid signal
  input  logic signed [W-1:0]  xi_q,      // input xi in Q5.26 format
  output logic                 valid_out, // output valid signal
  output logic signed [W-1:0]  s_xi_q     // s(xi) = K1*(xi + K2*xi³) in Q5.26
);

  // Constants in Q5.26 format
  localparam signed [W-1:0] K1 = 32'shF6CA89EF;  // -2·log2(e)·√(2/π) ≈ -2.30220819814
  localparam signed [W-1:0] K2 = 32'sh002DC6C5;  // 0.044715
  
  // Stage 1: Input capture & compute xi²
  logic signed [W-1:0]  xi_stage1;
  logic signed [63:0]   xi_sq_64_stage1;
  logic                 valid_stage1;
  
  // Stage 2: Scale xi² and compute xi³ = xi² × xi
  logic signed [W-1:0]  xi_stage2;
  logic signed [63:0]   xi_cube_64_stage2;
  logic                 valid_stage2;
  
  // Stage 3: Scale xi³ and compute K2 × xi³
  logic signed [W-1:0]  xi_stage3;
  logic signed [63:0]   k2_xi3_64_stage3;
  logic                 valid_stage3;
  
  // Stage 4: Scale K2×xi³ and compute sum = xi + K2×xi³
  logic signed [W-1:0]  sum_stage4;
  logic                 valid_stage4;
  
  // Stage 5: Compute K1 × sum
  logic signed [63:0]   final_64_stage5;
  logic                 valid_stage5;
  
  // Stage 6: Scale final result
  logic signed [W-1:0]  result_stage6;
  logic                 valid_stage6;
  
  always_ff @(posedge clk or negedge rst_n) begin
    if (!rst_n) begin
      // Stage 1
      xi_stage1         <= '0;
      xi_sq_64_stage1   <= 64'sd0;
      valid_stage1      <= 1'b0;
      
      // Stage 2
      xi_stage2         <= '0;
      xi_cube_64_stage2 <= 64'sd0;
      valid_stage2      <= 1'b0;
      
      // Stage 3
      xi_stage3         <= '0;
      k2_xi3_64_stage3  <= 64'sd0;
      valid_stage3      <= 1'b0;
      
      // Stage 4
      sum_stage4        <= '0;
      valid_stage4      <= 1'b0;
      
      // Stage 5
      final_64_stage5   <= 64'sd0;
      valid_stage5      <= 1'b0;
      
      // Stage 6 (Output)
      result_stage6     <= '0;
      valid_stage6      <= 1'b0;
      
    end else begin
      
      // Stage 1: Capture input AND compute xi²
      xi_stage1       <= xi_q;
      xi_sq_64_stage1 <= $signed(xi_q) * $signed(xi_q);
      valid_stage1    <= valid_in;
      
      // Stage 2: Scale xi² AND compute xi³ in same cycle
      // xi³ = (xi² >> Q) × xi
      xi_stage2         <= xi_stage1;
      xi_cube_64_stage2 <= ($signed(xi_sq_64_stage1) >>> Q) * $signed(xi_stage1);
      valid_stage2      <= valid_stage1;
      
      // Stage 3: Scale xi³ AND compute K2 × xi³ in same cycle
      // K2×xi³ = K2 × (xi³ >> Q)
      xi_stage3        <= xi_stage2;
      k2_xi3_64_stage3 <= $signed(K2) * ($signed(xi_cube_64_stage2) >>> Q);
      valid_stage3     <= valid_stage2;
      
      // Stage 4: Scale K2×xi³ AND compute sum in same cycle
      // sum = xi + (K2×xi³ >> Q)
      sum_stage4   <= $signed(xi_stage3) + ($signed(k2_xi3_64_stage3) >>> Q);
      valid_stage4 <= valid_stage3;
      
      // Stage 5: Compute K1 × sum
      final_64_stage5 <= $signed(K1) * $signed(sum_stage4);
      valid_stage5    <= valid_stage4;
      
      // Stage 6: Scale final result
      result_stage6 <= $signed(final_64_stage5) >>> Q;
      valid_stage6  <= valid_stage5;
    end
  end

  // Output assignment
  assign s_xi_q    = result_stage6;
  assign valid_out = valid_stage6;

endmodule
