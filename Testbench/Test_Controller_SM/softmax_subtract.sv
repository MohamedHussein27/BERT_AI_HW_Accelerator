//============================================================================
// Module:  softmax_subtract.sv
// Project: BERT Self-Attention Softmax Hardware Accelerator
// Description: Reads stored elements from the max module's SRAM and subtracts
//              the maximum value. Outputs a streaming (x_i - max) result,
//              one per clock cycle.
//============================================================================

`timescale 1ns/1ps

module softmax_subtract
  import softmax_pkg::*;
#(
  parameter int D_W   = DATA_W,
  parameter int IDX_W = SEQ_IDX_W
) (
  input  logic                      clk,
  input  logic                      rst_n,

  // Control
  input  logic                      start,
  input  logic [IDX_W-1:0]          vec_len_cfg,

  // Max value input (latched on start)
  input  logic signed [D_W-1:0]     max_value,

  // SRAM read port back to softmax_max
  output logic [IDX_W-1:0]          sram_rd_addr,
  input  logic signed [D_W-1:0]     sram_rd_data,

  // Output streaming interface
  output logic                      out_valid,
  output logic signed [D_W-1:0]     out_data,
  output logic                      out_last     // Pulses on last element
);

  //--------------------------------------------------------------------------
  // Registers
  //--------------------------------------------------------------------------
  logic                  active;
  logic [IDX_W-1:0]      cnt;
  logic [IDX_W-1:0]      vec_len_r;
  logic signed [D_W-1:0] max_r;

  // Pipeline: SRAM has 1-cycle read latency (registered output)
  logic                  pipe_valid;
  logic                  pipe_last;

  //--------------------------------------------------------------------------
  // SRAM address generation
  //--------------------------------------------------------------------------
  assign sram_rd_addr = cnt;

  //--------------------------------------------------------------------------
  // Main process — address generation
  //--------------------------------------------------------------------------
  always_ff @(posedge clk or negedge rst_n) begin
    if (!rst_n) begin
      active    <= 1'b0;
      cnt       <= '0;
      vec_len_r <= '0;
      max_r     <= '0;
    end else begin
      if (start && !active) begin
        active    <= 1'b1;
        cnt       <= '0;
        vec_len_r <= vec_len_cfg;
        max_r     <= max_value;
      end else if (active) begin
        if (cnt == vec_len_r - 1)
          active <= 1'b0;
        cnt <= cnt + 1'b1;
      end
    end
  end

  //--------------------------------------------------------------------------
  // Pipeline stage 1: read latency compensation
  //--------------------------------------------------------------------------
  always_ff @(posedge clk or negedge rst_n) begin
    if (!rst_n) begin
      pipe_valid <= 1'b0;
      pipe_last  <= 1'b0;
    end else begin
      pipe_valid <= active;
      pipe_last  <= active && (cnt == vec_len_r - 1);
    end
  end

  //--------------------------------------------------------------------------
  // Subtraction (combinational, registered output)
  //--------------------------------------------------------------------------
  always_ff @(posedge clk or negedge rst_n) begin
    if (!rst_n) begin
      out_valid <= 1'b0;
      out_data  <= '0;
      out_last  <= 1'b0;
    end else begin
      out_valid <= pipe_valid;
      out_last  <= pipe_last;
      if (pipe_valid)
        out_data <= $signed(sram_rd_data) - $signed(max_r);
      else
        out_data <= '0;
    end
  end

endmodule
