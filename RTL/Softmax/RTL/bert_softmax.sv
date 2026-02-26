//============================================================================
// Module:  bert_softmax.sv
// Project: BERT Self-Attention Softmax Hardware Accelerator
// Description: Top-level integration module for numerically stable,
//              FPGA-efficient Softmax computation.
//
//              Dataflow:
//              Input → Max → Subtract → Exp_PLA → Accumulate → Reciprocal
//                      → Normalize → Output
//
//              Interface: Valid/Ready handshake on input and output.
//              Parameterizable vector length (default 64) and data width.
//
//              For BERT self-attention: Softmax(QK^T / sqrt(d_k))
//============================================================================

`timescale 1ns/1ps

module bert_softmax
  import softmax_pkg::*;
#(
  parameter int VEC_LEN  = SEQ_LEN,       // Default 64
  parameter int D_W      = DATA_W,        // Input width Q5.26 (32)
  parameter int O_W      = NORM_W,        // Output width Q1.15 (16)
  parameter int MAX_LEN  = SEQ_LEN_MAX,   // Max supported vector length (128)
  parameter int IDX_W    = SEQ_IDX_W      // Index width
) (
  input  logic                   clk,
  input  logic                   rst_n,

  // Control
  input  logic                   start,         // Pulse to begin
  input  logic [IDX_W-1:0]       vec_len_cfg,   // Runtime vector length

  // Input streaming interface (valid/ready handshake)
  input  logic                   in_valid,
  output logic                   in_ready,
  input  logic signed [D_W-1:0]  in_data,       // Q5.26 signed

  // Output streaming interface
  output logic                   out_valid,
  output logic [O_W-1:0]         out_data,      // Q1.15 unsigned
  output logic                   out_last,       // Pulses on last output

  // Status
  output logic                   busy,
  output logic                   done
);

  //==========================================================================
  // Internal wires
  //==========================================================================

  // ---- Max module ----
  logic                  max_start;
  logic                  max_done;
  logic signed [D_W-1:0] max_value;
  logic [IDX_W-1:0]      max_rd_addr;
  logic signed [D_W-1:0] max_rd_data;
  logic                  max_in_ready;

  // ---- Subtract module ----
  logic                  sub_start;
  logic                  sub_out_valid;
  logic signed [D_W-1:0] sub_out_data;
  logic                  sub_out_last;

  // ---- Exp PLA module ----
  logic                  exp_out_valid;
  logic [EXP_W-1:0]      exp_out_data;

  // ---- Accumulator module ----
  logic                  acc_start;
  logic                  acc_sum_valid;
  logic [ACC_W-1:0]      acc_sum_out;
  logic [IDX_W-1:0]      acc_rd_addr;
  logic [EXP_W-1:0]      acc_rd_data;

  // ---- Reciprocal module ----
  logic                  recip_start;
  logic                  recip_done;
  logic [ACC_W-1:0]      recip_out;

  // ---- Normalize module ----
  logic                  norm_start;
  logic                  norm_out_valid;
  logic [O_W-1:0]        norm_out_data;
  logic                  norm_out_last;
  logic [IDX_W-1:0]      norm_exp_addr;
  logic [EXP_W-1:0]      norm_exp_data;

  // ---- FSM ----
  logic                  fsm_in_ready;
  sm_state_t             fsm_state;

  //==========================================================================
  // Input handshake
  //==========================================================================
  assign in_ready = max_in_ready && fsm_in_ready;

  //==========================================================================
  // Output connections
  //==========================================================================
  assign out_valid = norm_out_valid;
  assign out_data  = norm_out_data;
  assign out_last  = norm_out_last;

  //==========================================================================
  // SRAM multiplexing for accumulator read port
  //==========================================================================
  // During NORMALIZE phase, the normalize module reads from accumulator SRAM
  // Only normalize module reads exp SRAM
  assign acc_rd_addr = norm_exp_addr;
  assign norm_exp_data = acc_rd_data;

  //==========================================================================
  // Submodule Instantiations
  //==========================================================================

  // ---- Control FSM ----
  softmax_control_fsm #(
    .IDX_W      (IDX_W)
  ) u_fsm (
    .clk            (clk),
    .rst_n          (rst_n),
    .start          (start),
    .vec_len_cfg    (vec_len_cfg),
    .in_valid       (in_valid),
    .in_ready       (fsm_in_ready),
    .max_done       (max_done),
    .sub_exp_last   (sub_out_last),
    .acc_sum_valid  (acc_sum_valid),
    .recip_done     (recip_done),
    .norm_last      (norm_out_last),
    .max_start      (max_start),
    .sub_start      (sub_start),
    .acc_start      (acc_start),
    .recip_start    (recip_start),
    .norm_start     (norm_start),
    .busy           (busy),
    .done           (done),
    .current_state  (fsm_state)
  );

  // ---- Max Finder ----
  softmax_max #(
    .VEC_LEN    (VEC_LEN),
    .D_W        (D_W),
    .MAX_LEN    (MAX_LEN),
    .IDX_W      (IDX_W)
  ) u_max (
    .clk            (clk),
    .rst_n          (rst_n),
    .start          (max_start),
    .vec_len_cfg    (vec_len_cfg),
    .in_valid       (in_valid && fsm_in_ready),
    .in_ready       (max_in_ready),
    .in_data        (in_data),
    .max_valid      (max_done),
    .max_value      (max_value),
    .rd_addr        (max_rd_addr),
    .rd_data        (max_rd_data)
  );

  // ---- Subtract ----
  softmax_subtract #(
    .D_W        (D_W),
    .IDX_W      (IDX_W)
  ) u_subtract (
    .clk            (clk),
    .rst_n          (rst_n),
    .start          (sub_start),
    .vec_len_cfg    (vec_len_cfg),
    .max_value      (max_value),
    .sram_rd_addr   (max_rd_addr),
    .sram_rd_data   (max_rd_data),
    .out_valid      (sub_out_valid),
    .out_data       (sub_out_data),
    .out_last       (sub_out_last)
  );

  // ---- Exp PLA ----
  softmax_exp_pla #(
    .D_W        (D_W),
    .FRAC_I     (FRAC_IN),
    .O_W        (EXP_W),
    .FRAC_O     (FRAC_EXP),
    .NSEG       (PLA_NSEG),
    .COEFF_W    (PLA_COEFF_W),
    .COEFF_F    (PLA_COEFF_F),
    .H_SHIFT    (PLA_H_SHIFT)
  ) u_exp_pla (
    .clk            (clk),
    .rst_n          (rst_n),
    .in_valid       (sub_out_valid),
    .in_data        (sub_out_data),
    .out_valid      (exp_out_valid),
    .out_data       (exp_out_data)
  );

  // ---- Accumulator ----
  softmax_accumulator #(
    .I_W        (EXP_W),
    .FRAC_I     (FRAC_EXP),
    .O_W        (ACC_W),
    .FRAC_O     (FRAC_ACC),
    .MAX_LEN    (MAX_LEN),
    .IDX_W      (IDX_W)
  ) u_accumulator (
    .clk            (clk),
    .rst_n          (rst_n),
    .start          (acc_start),
    .vec_len_cfg    (vec_len_cfg),
    .in_valid       (exp_out_valid),
    .in_data        (exp_out_data),
    .sum_valid      (acc_sum_valid),
    .sum_out        (acc_sum_out),
    .rd_addr        (acc_rd_addr),
    .rd_data        (acc_rd_data)
  );

  // ---- Reciprocal ----
  softmax_reciprocal #(
    .W          (ACC_W),
    .Q          (FRAC_ACC),
    .LUT_BITS   (NR_LUT_BITS),
    .ITER       (NR_ITER)
  ) u_reciprocal (
    .clk            (clk),
    .rst_n          (rst_n),
    .in_valid       (recip_start),
    .in_data        (acc_sum_out),
    .out_valid      (recip_done),
    .out_data       (recip_out)
  );

  // ---- Normalize ----
  softmax_normalize #(
    .EXP_WIDTH  (EXP_W),
    .FRAC_E     (FRAC_EXP),
    .RCP_WIDTH  (ACC_W),
    .FRAC_R     (FRAC_ACC),
    .OUT_WIDTH  (O_W),
    .FRAC_OUT   (FRAC_NORM),
    .MAX_LEN    (MAX_LEN),
    .IDX_W      (IDX_W)
  ) u_normalize (
    .clk            (clk),
    .rst_n          (rst_n),
    .start          (norm_start),
    .vec_len_cfg    (vec_len_cfg),
    .reciprocal     (recip_out),
    .exp_rd_addr    (norm_exp_addr),
    .exp_rd_data    (norm_exp_data),
    .out_valid      (norm_out_valid),
    .out_data       (norm_out_data),
    .out_last       (norm_out_last)
  );

endmodule
