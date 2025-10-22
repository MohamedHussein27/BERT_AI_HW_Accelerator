`timescale 1ns/1ps
// ============================================================================
// Single GELU Lane - Complete Corrected Architecture
// Flow: Polynomial(6) → EU1(3) → DU(4) → EU2(3) → Output(16)
// ============================================================================
module GELU_Lane #(
  parameter int Q = 26,
  parameter int W = 32,
  parameter int INT_WIDTH = 5
) (
  input  logic                     clk,
  input  logic                     rst_n,
  input  logic                     valid_in,
  input  logic signed [W-1:0]      xi,            // input value (F)
  
  // LUT interface for EU1 (first exponential)
  output logic [2:0]               segment_index_1,
  input  logic signed [W-1:0]      k_coeff_1,
  input  logic signed [W-1:0]      b_intercept_1,
  
  // LUT interface for EU2 (second exponential)
  output logic [2:0]               segment_index_2,
  input  logic signed [W-1:0]      k_coeff_2,
  input  logic signed [W-1:0]      b_intercept_2,
  
  output logic                     valid_out,
  output logic signed [W-1:0]      gelu_result,
  output logic                     div_by_zero
);

  // ============================================================================
  // Shift Register for xi (Delay = Polynomial + EU1 = 6 + 3 = 9 cycles)
  // ============================================================================
  localparam int PIPELINE_DELAY = 9;
  logic signed [W-1:0] xi_pipe [0:PIPELINE_DELAY-1];
  logic signed [W-1:0] xi_delayed;
  
  always_ff @(posedge clk or negedge rst_n) begin
    if (!rst_n) begin
      for (int i = 0; i < PIPELINE_DELAY; i++) 
        xi_pipe[i] <= '0;
    end else begin
      xi_pipe[0] <= xi;
      for (int i = 1; i < PIPELINE_DELAY; i++)
        xi_pipe[i] <= xi_pipe[i-1];
    end
  end
  
  assign xi_delayed = xi_pipe[PIPELINE_DELAY-1];

  // ============================================================================
  // Stage 1: Polynomial Unit (6 cycles)
  // ============================================================================
  logic signed [W-1:0] s_xi;
  logic                poly_valid;
  
  PolynomialUnit #(
    .Q(Q),
    .W(W)
  ) poly_inst (
    .clk(clk),
    .rst_n(rst_n),
    .valid_in(valid_in),
    .xi_q(xi),
    .valid_out(poly_valid),
    .s_xi_q(s_xi)
  );

  // ============================================================================
  // Stage 2: First Exponential Unit - EU1 (3 cycles)
  // ============================================================================
  logic [INT_WIDTH-1:0] s_xi_integer;
  logic [Q-1:0]         s_xi_frac;
  logic signed [W-1:0]  exp_s_xi;
  logic                 eu1_valid;
  
  assign s_xi_integer = s_xi[W-1:Q];
  assign s_xi_frac    = s_xi[Q-1:0];
  assign segment_index_1 = s_xi_frac[Q-1:Q-3];  // bits [25:23]

  EU #(
    .Q(Q),
    .W(W),
    .INT_WIDTH(INT_WIDTH)
  ) eu1_inst (
    .clk(clk),
    .rst_n(rst_n),
    .valid_in(poly_valid),
    .integer_part(s_xi_integer),
    .frac_part(s_xi_frac),
    .k_coeff(k_coeff_1),
    .b_intercept(b_intercept_1),
    .valid_out(eu1_valid),
    .exp_result(exp_s_xi)
  );

  // ============================================================================
  // Stage 3: Division Unit (DU)
  // ============================================================================
  logic signed [W-1:0] division_result;
  logic                du_valid;
  logic                result_sign;
  
  DU #(
    .Q(Q),
    .W(W)
  ) du_inst (
    .clk(clk),
    .rst_n(rst_n),
    .valid_in(eu1_valid),
    .F(xi_delayed),           // Original input (delayed by 9 cycles)
    .s_xi(exp_s_xi),          // exp(s_xi) from EU1
    .valid_out(du_valid),
    .exponent(division_result),
    .result_sign(result_sign)
  );

  // ============================================================================
  // Zero Detection Logic (combinational)
  // ============================================================================
  logic zero_numerator_s0, zero_denominator_s0;
  always_comb begin
    zero_numerator_s0   = (xi_delayed == 0);
    zero_denominator_s0 = ((1 + exp_s_xi) == 0);
  end

  // ============================================================================
  // Pipelining Zero Flags (Delay = DU + EU1 = 7 cycles)
  // ============================================================================
  localparam int TOTAL_DELAY = 7;
  logic [TOTAL_DELAY-1:0] zero_num_pipe;
  logic [TOTAL_DELAY-1:0] zero_denom_pipe;

  always_ff @(posedge clk or negedge rst_n) begin
    if (!rst_n) begin
      zero_num_pipe   <= '0;
      zero_denom_pipe <= '0;
    end else begin
      zero_num_pipe[0]   <= zero_numerator_s0;
      zero_denom_pipe[0] <= zero_denominator_s0;
      for (int i = 1; i < TOTAL_DELAY; i++) begin
        zero_num_pipe[i]   <= zero_num_pipe[i-1];
        zero_denom_pipe[i] <= zero_denom_pipe[i-1];
      end
    end
  end

  logic zero_num_final, zero_denom_final;
  assign zero_num_final   = zero_num_pipe[TOTAL_DELAY-1];
  assign zero_denom_final = zero_denom_pipe[TOTAL_DELAY-1];

  // ============================================================================
  // Stage 4: Second Exponential Unit - EU2 (3 cycles)
  // ============================================================================
  logic [INT_WIDTH-1:0] div_integer;
  logic [Q-1:0]         div_frac;
  logic signed [W-1:0]  final_result;
  logic                 eu2_valid;
  
  assign div_integer = division_result[W-1:Q];
  assign div_frac    = division_result[Q-1:0];
  assign segment_index_2 = div_frac[Q-1:Q-3];

  EU #(
    .Q(Q),
    .W(W),
    .INT_WIDTH(INT_WIDTH)
  ) eu2_inst (
    .clk(clk),
    .rst_n(rst_n),
    .valid_in(du_valid),
    .integer_part(div_integer),
    .frac_part(div_frac),
    .k_coeff(k_coeff_2),
    .b_intercept(b_intercept_2),
    .valid_out(eu2_valid),
    .exp_result(final_result)
  );

  // ============================================================================
  // Stage 5: Final Output with Zero Handling & Sign Correction
  // ============================================================================
  always_ff @(posedge clk or negedge rst_n) begin
    if (!rst_n) begin
      gelu_result  <= '0;
      valid_out    <= 1'b0;
      div_by_zero  <= 1'b0;
    end else begin
      valid_out   <= eu2_valid;
      div_by_zero <= zero_denom_final;

      if (eu2_valid) begin
        if (zero_denom_final || zero_num_final) begin
          gelu_result <= '0;            // Numerator zero → 0
        end else begin
          gelu_result <= result_sign ? -final_result : final_result;
        end
      end
    end
  end

endmodule
