//============================================================================
// Module:  softmax_max.sv
// Project: BERT Self-Attention Softmax Hardware Accelerator
// Description: Streaming row-max finder. Accepts one signed Q5.26 element
//              per clock cycle, stores all inputs in a dual-port SRAM, and
//              outputs the row maximum after VEC_LEN elements.
//============================================================================

`timescale 1ns/1ps

module softmax_max
  import softmax_pkg::*;
#(
  parameter int VEC_LEN  = SEQ_LEN,
  parameter int D_W      = DATA_W,
  parameter int MAX_LEN  = SEQ_LEN_MAX,
  parameter int IDX_W    = SEQ_IDX_W
) (
  input  logic                      clk,
  input  logic                      rst_n,

  // Control
  input  logic                      start,        // Pulse to begin accepting data
  input  logic [IDX_W-1:0]          vec_len_cfg,  // Runtime vector length (up to MAX_LEN)

  // Input streaming interface
  input  logic                      in_valid,
  output logic                      in_ready,
  input  logic signed [D_W-1:0]     in_data,

  // Output: max value (valid for 1 cycle when done)
  output logic                      max_valid,
  output logic signed [D_W-1:0]     max_value,

  // SRAM read port (for subtract phase)
  input  logic [IDX_W-1:0]          rd_addr,
  output logic signed [D_W-1:0]     rd_data
);

  //--------------------------------------------------------------------------
  // Internal SRAM for storing input vector
  //--------------------------------------------------------------------------
  logic signed [D_W-1:0] sram [0:MAX_LEN-1];

  //--------------------------------------------------------------------------
  // Registers
  //--------------------------------------------------------------------------
  logic                  active;
  logic [IDX_W-1:0]      cnt;
  logic [IDX_W-1:0]      vec_len_r;
  logic signed [D_W-1:0] running_max;

  //--------------------------------------------------------------------------
  // SRAM read port (synchronous read for pipeline alignment)
  //--------------------------------------------------------------------------
  always_ff @(posedge clk) begin
    rd_data <= sram[rd_addr];
  end

  //--------------------------------------------------------------------------
  // Input handshake
  //--------------------------------------------------------------------------
  assign in_ready = active;

  //--------------------------------------------------------------------------
  // Main process
  //--------------------------------------------------------------------------
  always_ff @(posedge clk or negedge rst_n) begin
    if (!rst_n) begin
      active      <= 1'b0;
      cnt         <= '0;
      vec_len_r   <= '0;
      running_max <= {1'b1, {(D_W-1){1'b0}}}; // Most negative value
      max_valid   <= 1'b0;
      max_value   <= '0;
    end else begin
      max_valid <= 1'b0; // Default: single-cycle pulse

      if (start && !active) begin
        active      <= 1'b1;
        cnt         <= '0;
        vec_len_r   <= vec_len_cfg;
        running_max <= {1'b1, {(D_W-1){1'b0}}}; // Reset to most-negative
      end else if (active && in_valid) begin
        // Store input to SRAM
        sram[cnt] <= in_data;

        // Update running max
        if ($signed(in_data) > $signed(running_max))
          running_max <= in_data;

        // Check if last element
        if (cnt == vec_len_r - 1) begin
          active    <= 1'b0;
          max_valid <= 1'b1;
          // Output max (use combinational compare for final element)
          if ($signed(in_data) > $signed(running_max))
            max_value <= in_data;
          else
            max_value <= running_max;
        end

        cnt <= cnt + 1'b1;
      end
    end
  end

endmodule
