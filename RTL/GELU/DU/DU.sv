`timescale 1ns/1ps
// Division Unit (DU) for GELU 
module DU #(
  parameter int Q = 26,      // fractional bits (Q5.26)
  parameter int W = 32       // data width (bits)
) (
  input  logic                 clk,
  input  logic                 rst_n,      // async active-low reset
  input  logic                 valid_in,   // input valid signal
  input  logic signed [W-1:0]  F,          // numerator
  input  logic signed [W-1:0]  s_xi,       // denominator
  output logic                 valid_out,  // output valid signal
  output logic signed [W-1:0]  exponent,   // (m1+s1) - (m2+s2)
  output logic                 div_by_zero,// flag
  output logic                 result_sign // sign of result (F_sign ^ s_xi_sign)
);

  // Constant for Q in proper width
  localparam logic signed [W-1:0] Q_CONST = W'(signed'(Q));

  // Stage 1: absolute values and signs
  logic [W-1:0]        F_abs_stage1;
  logic [W-1:0]        s_xi_abs_stage1;
  logic                F_sign_stage1;
  logic                s_xi_sign_stage1;
  logic                valid_stage1;

  // Stage 2: LOD outputs
  logic [$clog2(W)-1:0] F_lod_pos;
  logic [$clog2(W)-1:0] sxi_lod_pos;
  logic                 F_found;
  logic                 sxi_found;
  logic [W-1:0]         F_abs_stage2;
  logic [W-1:0]         s_xi_abs_stage2;
  logic                 F_sign_stage2;
  logic                 s_xi_sign_stage2;
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
  logic                div_zero_stage4;
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

  LOD #(.W(W)) lod_sxi (
    .clk(clk),
    .rst_n(rst_n),
    .valid_in(valid_stage1),
    .data_in(s_xi_abs_stage1),
    .valid_out(lod_valid_2),
    .lod_pos(sxi_lod_pos),
    .found(sxi_found)
  );

  //  Pipeline
  always_ff @(posedge clk or negedge rst_n) begin
    if (!rst_n) begin
      // Stage 1
      F_abs_stage1     <= '0;
      s_xi_abs_stage1  <= '0;
      F_sign_stage1    <= 1'b0;
      s_xi_sign_stage1 <= 1'b0;
      valid_stage1     <= 1'b0;

      // Stage 2
      F_abs_stage2     <= '0;
      s_xi_abs_stage2  <= '0;
      F_sign_stage2    <= 1'b0;
      s_xi_sign_stage2 <= 1'b0;

      // Stage 3
      m1_stage3        <= '0;
      m2_stage3        <= '0;
      s1_stage3        <= '0;
      s2_stage3        <= '0;
      result_sign_stage3 <= 1'b0;
      valid_stage3     <= 1'b0;

      // Stage 4
      exponent_stage4  <= '0;
      div_zero_stage4  <= 1'b0;
      valid_stage4     <= 1'b0;
      result_sign_stage4 <= 1'b0;

    end else begin
      // ===== Stage 1 =====
      F_sign_stage1    <= F[W-1];
      s_xi_sign_stage1 <= s_xi[W-1];
      F_abs_stage1     <= F[W-1] ? -F : F;
      s_xi_abs_stage1  <= s_xi[W-1] ? -s_xi : s_xi;
      valid_stage1     <= valid_in;

      // ===== Stage 2 =====
      F_abs_stage2     <= F_abs_stage1;
      s_xi_abs_stage2  <= s_xi_abs_stage1;
      F_sign_stage2    <= F_sign_stage1;
      s_xi_sign_stage2 <= s_xi_sign_stage1;

      // ===== Stage 3 =====
      if (F_found) begin
        // Calculate exponent: s1 = lod_pos - Q
        // Extend lod_pos to full width as signed, then subtract
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

      if (sxi_found) begin
        // Calculate exponent: s2 = lod_pos - Q
        // Extend lod_pos to full width as signed, then subtract
        s2_stage3 <= ($signed({1'b0, {(W-$clog2(W)-1){1'b0}}, sxi_lod_pos}) - Q_CONST) << Q;        
        if (sxi_lod_pos > Q)
          m2_stage3 <= signed'(s_xi_abs_stage2 >> (sxi_lod_pos - Q));
        else if (sxi_lod_pos < Q)
          m2_stage3 <= signed'(s_xi_abs_stage2 << (Q - sxi_lod_pos));
        else
          m2_stage3 <= signed'(s_xi_abs_stage2);
      end else begin
        m2_stage3 <= '0;
        s2_stage3 <= '0;
      end

      // sign and valid for stage3
      result_sign_stage3 <= F_sign_stage2 ^ s_xi_sign_stage2;
      valid_stage3       <= lod_valid_1 && lod_valid_2;

      // ===== Stage 4: FINAL OUTPUT =====
      result_sign_stage4 <= result_sign_stage3;

      if (!sxi_found && valid_stage3) begin
        // Division by zero handling: exponent=0, assert div_by_zero
        exponent_stage4 <= '0;
        div_zero_stage4 <= 1'b1;
      end else begin
        exponent_stage4 <= (m1_stage3 + s1_stage3) - (m2_stage3 + s2_stage3);
        div_zero_stage4 <= 1'b0;
      end

      valid_stage4 <= valid_stage3;
    end
  end

  //  Final outputs
  assign exponent    = exponent_stage4;
  assign valid_out   = valid_stage4;
  assign div_by_zero = div_zero_stage4;
  assign result_sign = result_sign_stage4;

endmodule
