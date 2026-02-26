//============================================================================
// Module:  softmax_accumulator.sv
// Project: BERT Self-Attention Softmax Hardware Accelerator
// Description: Accumulates exp(x_i - max) values into a running sum.
//              Stores individual exp values into SRAM for the normalize pass.
//              Outputs the total sum after VEC_LEN elements.
//
//              Input:  Q1.15 unsigned (16-bit) exp values
//              Output: Q8.24 unsigned (32-bit) accumulated sum
//              SRAM:   Stores Q1.15 exp values for later normalization
//============================================================================

`timescale 1ns/1ps

module softmax_accumulator
  import softmax_pkg::*;
#(
  parameter int I_W     = EXP_W,         // Input width (16)
  parameter int FRAC_I  = FRAC_EXP,      // Input frac bits (15)
  parameter int O_W     = ACC_W,         // Output/accumulator width (32)
  parameter int FRAC_O  = FRAC_ACC,      // Output frac bits (24)
  parameter int MAX_LEN = SEQ_LEN_MAX,
  parameter int IDX_W   = SEQ_IDX_W
) (
  input  logic                  clk,
  input  logic                  rst_n,

  // Control
  input  logic                  start,
  input  logic [IDX_W-1:0]     vec_len_cfg,

  // Streaming input (from exp_pla)
  input  logic                  in_valid,
  input  logic [I_W-1:0]       in_data,    // Q1.15 unsigned

  // Output: accumulated sum
  output logic                  sum_valid,
  output logic [O_W-1:0]       sum_out,    // Q8.24 unsigned

  // SRAM read port (for normalize phase)
  input  logic [IDX_W-1:0]     rd_addr,
  output logic [I_W-1:0]       rd_data
);

  //--------------------------------------------------------------------------
  // Internal SRAM for storing exp values
  // Uses distributed RAM (async read is needed by normalize pipeline)
  //--------------------------------------------------------------------------
  (* ram_style = "distributed" *) logic [I_W-1:0] exp_sram [0:MAX_LEN-1];

  //--------------------------------------------------------------------------
  // Registers
  //--------------------------------------------------------------------------
  logic                active;
  logic [IDX_W-1:0]    cnt;
  logic [IDX_W-1:0]    vec_len_r;
  logic [O_W-1:0]      acc;

  //--------------------------------------------------------------------------
  // SRAM read port (combinational — distributed RAM)
  //--------------------------------------------------------------------------
  assign rd_data = exp_sram[rd_addr];

  //--------------------------------------------------------------------------
  // SRAM write port (separate from async-reset block for clean synthesis)
  //--------------------------------------------------------------------------
  always_ff @(posedge clk) begin
    if (active && in_valid)
      exp_sram[cnt] <= in_data;
  end

  //--------------------------------------------------------------------------
  // Accumulation
  //--------------------------------------------------------------------------

  // Align input Q1.15 to accumulator Q8.24:
  // Shift left by (FRAC_O - FRAC_I) = 24 - 15 = 9 bits
  localparam int ALIGN_SHIFT = FRAC_O - FRAC_I;

  logic [O_W-1:0] in_aligned;
  assign in_aligned = {{(O_W - I_W - ALIGN_SHIFT){1'b0}}, in_data, {ALIGN_SHIFT{1'b0}}};

  always_ff @(posedge clk or negedge rst_n) begin
    if (!rst_n) begin
      active    <= 1'b0;
      cnt       <= '0;
      vec_len_r <= '0;
      acc       <= '0;
      sum_valid <= 1'b0;
      sum_out   <= '0;
    end else begin
      sum_valid <= 1'b0;

      if (start) begin
        active    <= 1'b1;
        cnt       <= '0;
        vec_len_r <= vec_len_cfg;
        acc       <= '0;
      end else if (active && in_valid) begin
        // Accumulate (align and add)
        acc <= acc + in_aligned;

        // Check if last element
        if (cnt == vec_len_r - 1) begin
          active    <= 1'b0;
          sum_valid <= 1'b1;
          sum_out   <= acc + in_aligned;
        end

        cnt <= cnt + 1'b1;
      end
    end
  end

endmodule
