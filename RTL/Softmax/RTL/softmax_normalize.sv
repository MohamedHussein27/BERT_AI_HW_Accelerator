//============================================================================
// Module:  softmax_normalize.sv
// Project: BERT Self-Attention Softmax Hardware Accelerator
// Description: Reads stored exp values from accumulator SRAM, multiplies
//              each by the reciprocal of the sum, and outputs streaming
//              normalized softmax values.
//
//              Operation: output_i = exp_i * reciprocal(sum)
//              Input:  exp_i is Q1.15 (16-bit), reciprocal is Q8.24 (32-bit)
//              Output: Q1.15 unsigned (16-bit) normalized softmax
//
//              Pipeline: 2 cycles (multiply + shift/truncate)
//============================================================================

`timescale 1ns/1ps

module softmax_normalize
  import softmax_pkg::*;
#(
  parameter int EXP_WIDTH  = EXP_W,     // 16
  parameter int FRAC_E     = FRAC_EXP,  // 15
  parameter int RCP_WIDTH  = ACC_W,     // 32
  parameter int FRAC_R     = FRAC_ACC,  // 24
  parameter int OUT_WIDTH  = NORM_W,    // 16
  parameter int FRAC_OUT   = FRAC_NORM, // 15
  parameter int MAX_LEN    = SEQ_LEN_MAX,
  parameter int IDX_W      = SEQ_IDX_W
) (
  input  logic                      clk,
  input  logic                      rst_n,

  // Control
  input  logic                      start,
  input  logic [IDX_W-1:0]          vec_len_cfg,

  // Reciprocal input (latched on start)
  input  logic [RCP_WIDTH-1:0]      reciprocal,  // Q8.24

  // SRAM read port to accumulator's exp SRAM
  output logic [IDX_W-1:0]          exp_rd_addr,
  input  logic [EXP_WIDTH-1:0]      exp_rd_data, // Q1.15

  // Output streaming interface
  output logic                      out_valid,
  output logic [OUT_WIDTH-1:0]      out_data,    // Q1.15 unsigned
  output logic                      out_last
);

  //--------------------------------------------------------------------------
  // Registers
  //--------------------------------------------------------------------------
  logic                    active;
  logic [IDX_W-1:0]        cnt;
  logic [IDX_W-1:0]        vec_len_r;
  logic [RCP_WIDTH-1:0]    recip_r;

  //--------------------------------------------------------------------------
  // Pipeline signals
  //--------------------------------------------------------------------------
  // Stage 0: Address generation (combinational)
  // Stage 1: SRAM read latency + multiply
  // Stage 2: Shift + truncate

  logic                    p1_valid;
  logic                    p1_last;
  logic [EXP_WIDTH-1:0]    p1_exp;

  logic                    p2_valid;
  logic                    p2_last;

  //--------------------------------------------------------------------------
  // Address generation
  //--------------------------------------------------------------------------
  assign exp_rd_addr = cnt;

  always_ff @(posedge clk or negedge rst_n) begin
    if (!rst_n) begin
      active    <= 1'b0;
      cnt       <= '0;
      vec_len_r <= '0;
      recip_r   <= '0;
    end else begin
      if (start && !active) begin
        active    <= 1'b1;
        cnt       <= '0;
        vec_len_r <= vec_len_cfg;
        recip_r   <= reciprocal;
      end else if (active) begin
        if (cnt == vec_len_r - 1)
          active <= 1'b0;
        cnt <= cnt + 1'b1;
      end
    end
  end

  //--------------------------------------------------------------------------
  // Pipeline Stage 1: SRAM read + capture
  //--------------------------------------------------------------------------
  always_ff @(posedge clk or negedge rst_n) begin
    if (!rst_n) begin
      p1_valid <= 1'b0;
      p1_last  <= 1'b0;
      p1_exp   <= '0;
    end else begin
      p1_valid <= active;
      p1_last  <= active && (cnt == vec_len_r - 1);
      p1_exp   <= exp_rd_data;
    end
  end

  //--------------------------------------------------------------------------
  // Pipeline Stage 2: Multiply + Shift + Output
  //--------------------------------------------------------------------------
  // exp_i (Q1.15 unsigned, 16b) * reciprocal (Q8.24 unsigned, 32b)
  // Product: Q(1+8).(15+24) = Q9.39, 48 bits
  // We need Q1.15 output: shift right by (39 - 15) = 24 bits

  localparam int PROD_W    = EXP_WIDTH + RCP_WIDTH;  // 48
  localparam int PROD_FRAC = FRAC_E + FRAC_R;        // 39
  localparam int OUT_SHIFT = PROD_FRAC - FRAC_OUT;   // 24

  logic [PROD_W-1:0] product;
  // CRITICAL: extend both operands to PROD_W before multiply
  // In Verilog, 16b * 32b = 32b (max of operand widths), NOT 48b!
  logic [PROD_W-1:0] exp_ext;
  logic [PROD_W-1:0] rcp_ext;
  assign exp_ext = {{(PROD_W-EXP_WIDTH){1'b0}}, p1_exp};
  assign rcp_ext = {{(PROD_W-RCP_WIDTH){1'b0}}, recip_r};
  assign product = exp_ext * rcp_ext;

  always_ff @(posedge clk or negedge rst_n) begin
    if (!rst_n) begin
      p2_valid <= 1'b0;
      p2_last  <= 1'b0;
      out_data <= '0;
    end else begin
      p2_valid <= p1_valid;
      p2_last  <= p1_last;

      if (p1_valid) begin
        // Shift and truncate to Q1.15
        // Saturate if needed (shouldn't happen for softmax outputs)
        if (product[PROD_W-1 : OUT_SHIFT + OUT_WIDTH] != '0)
          out_data <= {OUT_WIDTH{1'b1}}; // Saturate
        else
          out_data <= product[OUT_SHIFT +: OUT_WIDTH];
      end else begin
        out_data <= '0;
      end
    end
  end

  assign out_valid = p2_valid;
  assign out_last  = p2_last;

endmodule
