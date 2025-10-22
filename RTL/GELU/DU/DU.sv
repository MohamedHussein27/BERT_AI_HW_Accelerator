`timescale 1ns/1ps
// Division Unit (DU) for GELU 
// Computes: F / (1 + s_xi)
module DU #(
  parameter int Q = 26,      // fractional bits (Q5.26)
  parameter int W = 32       // data width (bits)
) (
  input  logic                 clk,
  input  logic                 rst_n,      // async active-low reset
  input  logic                 valid_in,   // input valid signal
  input  logic signed [W-1:0]  F,          // numerator
  input  logic signed [W-1:0]  s_xi,       // exp(s_xi) - will add 1 to this
  output logic                 valid_out,  // output valid signal
  output logic signed [W-1:0]  exponent,   // (m1+s1) - (m2+s2)
  output logic                 result_sign // sign of result (F_sign ^ denominator_sign)
);

  // Constant for +1 in Q5.26 format
  localparam logic signed [W-1:0] ONE_Q = 32'h04000000;  // 1.0 = 2^26
  localparam logic signed [W-1:0] Q_CONST = W'(signed'(Q));

  // Stage 0: Add 1 to denominator
  logic signed [W-1:0] F_stage0;
  logic signed [W-1:0] denominator_stage0;  // 1 + s_xi
  logic                valid_stage0;

  // Stage 1: absolute values and signs
  logic [W-1:0]        F_abs_stage1;
  logic [W-1:0]        denom_abs_stage1;
  logic                F_sign_stage1;
  logic                denom_sign_stage1;
  logic                valid_stage1;

  // Stage 2: LOD outputs
  logic [$clog2(W)-1:0] F_lod_pos;
  logic [$clog2(W)-1:0] denom_lod_pos;
  logic                 F_found;
  logic                 denom_found;
  logic [W-1:0]         F_abs_stage2;
  logic [W-1:0]         denom_abs_stage2;
  logic                 F_sign_stage2;
  logic                 denom_sign_stage2;
  logic                 lod_valid_1, lod_valid_2;

  // Stage 3: normalized mantissas & exponents
  logic signed [W-1:0] m1_stage3;
  logic signed [W-1:0] m2_stage3;
  logic signed [W-1:0] s1_stage3;
  logic signed [W-1:0] s2_stage3;
  logic                result_sign_stage3;
  logic                valid_stage3;

  // Stage 4: final exponent difference + sign
  logic signed [W-1:0] exponent_stage4;
  logic                valid_stage4;
  logic                result_sign_stage4;

  //  LOD instances
  LOD #(.W(W)) lod_F (
    .clk(clk),
    .rst_n(rst_n),
    .valid_in(valid_stage1),
    .data_in(F_abs_stage1),
    .valid_out(lod_valid_1),
    .lod_pos(F_lod_pos),
    .found(F_found)
  );

  LOD #(.W(W)) lod_denom (
    .clk(clk),
    .rst_n(rst_n),
    .valid_in(valid_stage1),
    .data_in(denom_abs_stage1),
    .valid_out(lod_valid_2),
    .lod_pos(denom_lod_pos),
    .found(denom_found)
  );

  //  Pipeline
  always_ff @(posedge clk or negedge rst_n) begin
    if (!rst_n) begin
      // Stage 0
      F_stage0           <= '0;
      denominator_stage0 <= '0;
      valid_stage0       <= 1'b0;
      
      // Stage 1
      F_abs_stage1       <= '0;
      denom_abs_stage1   <= '0;
      F_sign_stage1      <= 1'b0;
      denom_sign_stage1  <= 1'b0;
      valid_stage1       <= 1'b0;

      // Stage 2
      F_abs_stage2       <= '0;
      denom_abs_stage2   <= '0;
      F_sign_stage2      <= 1'b0;
      denom_sign_stage2  <= 1'b0;

      // Stage 3
      m1_stage3          <= '0;
      m2_stage3          <= '0;
      s1_stage3          <= '0;
      s2_stage3          <= '0;
      result_sign_stage3 <= 1'b0;
      valid_stage3       <= 1'b0;

      // Stage 4
      exponent_stage4    <= '0;
      valid_stage4       <= 1'b0;
      result_sign_stage4 <= 1'b0;

    end else begin
      // ===== Stage 0: Add 1 to denominator =====
      F_stage0           <= F;
      denominator_stage0 <= ONE_Q + s_xi;  // CRITICAL: 1 + exp(s_xi)
      valid_stage0       <= valid_in;
      
      // ===== Stage 1: Compute absolute values =====
      F_sign_stage1      <= F_stage0[W-1];
      denom_sign_stage1  <= denominator_stage0[W-1];
      F_abs_stage1       <= F_stage0[W-1] ? -F_stage0 : F_stage0;
      denom_abs_stage1   <= denominator_stage0[W-1] ? -denominator_stage0 : denominator_stage0;
      valid_stage1       <= valid_stage0;

      // ===== Stage 2: Pass through for LOD =====
      F_abs_stage2       <= F_abs_stage1;
      denom_abs_stage2   <= denom_abs_stage1;
      F_sign_stage2      <= F_sign_stage1;
      denom_sign_stage2  <= denom_sign_stage1;

      // ===== Stage 3: Normalization =====
      if (F_found) begin
        s1_stage3 <= ($signed({1'b0, {(W-$clog2(W)-1){1'b0}}, F_lod_pos}) - Q_CONST) << Q;            
        if (F_lod_pos > Q)
          m1_stage3 <= signed'(F_abs_stage2 >> (F_lod_pos - Q));
        else if (F_lod_pos < Q)
          m1_stage3 <= signed'(F_abs_stage2 << (Q - F_lod_pos));
        else
          m1_stage3 <= signed'(F_abs_stage2);
      end else begin
        m1_stage3 <= '0;
        s1_stage3 <= '0;
      end

      if (denom_found) begin
        s2_stage3 <= ($signed({1'b0, {(W-$clog2(W)-1){1'b0}}, denom_lod_pos}) - Q_CONST) << Q;        
        if (denom_lod_pos > Q)
          m2_stage3 <= signed'(denom_abs_stage2 >> (denom_lod_pos - Q));
        else if (denom_lod_pos < Q)
          m2_stage3 <= signed'(denom_abs_stage2 << (Q - denom_lod_pos));
        else
          m2_stage3 <= signed'(denom_abs_stage2);
      end else begin
        m2_stage3 <= '0;
        s2_stage3 <= '0;
      end

      result_sign_stage3 <= F_sign_stage2 ^ denom_sign_stage2;
      valid_stage3       <= lod_valid_1 && lod_valid_2;

      // ===== Stage 4: FINAL OUTPUT =====
      exponent_stage4 <= (m1_stage3 + s1_stage3) - (m2_stage3 + s2_stage3);
      result_sign_stage4 <= result_sign_stage3;
      valid_stage4 <= valid_stage3;
    end
  end

  //  Final outputs
  assign exponent    = exponent_stage4;
  assign valid_out   = valid_stage4;
  assign result_sign = result_sign_stage4;

endmodule
