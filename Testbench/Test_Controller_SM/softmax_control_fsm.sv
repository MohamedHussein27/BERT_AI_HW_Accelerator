//============================================================================
// Module:  softmax_control_fsm.sv
// Project: BERT Self-Attention Softmax Hardware Accelerator
// Description: Master control FSM for the Softmax pipeline.
//              Orchestrates the multi-phase dataflow:
//                Phase 0: IDLE — wait for start
//                Phase 1: LOAD_MAX — stream inputs, find max, store in SRAM
//                Phase 2: SUB_EXP — subtract max, compute exp, accumulate sum
//                Phase 3: RECIPROCAL — compute 1/sum via Newton-Raphson
//                Phase 4: NORMALIZE — multiply exp values by reciprocal
//                Phase 5: DONE — signal completion
//============================================================================

`timescale 1ns/1ps

module softmax_control_fsm
  import softmax_pkg::*;
#(
  parameter int IDX_W = SEQ_IDX_W
) (
  input  logic                  clk,
  input  logic                  rst_n,

  // External interface
  input  logic                  start,          // Begin softmax computation
  input  logic [IDX_W-1:0]      vec_len_cfg,    // Vector length

  // handshake with input
  input  logic                  in_valid,
  output logic                  in_ready,

  // Status signals from submodules
  input  logic                  max_done,       // softmax_max completed
  input  logic                  sub_exp_last,   // Last exp value from pipeline
  input  logic                  acc_sum_valid,  // Accumulator sum ready
  input  logic                  recip_done,     // Reciprocal computed
  input  logic                  norm_last,      // Last normalized output

  // Control signals to submodules
  output logic                  max_start,
  output logic                  sub_start,
  output logic                  acc_start,
  output logic                  recip_start,
  output logic                  norm_start,

  // Status
  output logic                  busy,
  output logic                  done,
  output sm_state_t             current_state
);

  //--------------------------------------------------------------------------
  // State machine
  //--------------------------------------------------------------------------
  sm_state_t state, next_state;
  assign current_state = state;

  //--------------------------------------------------------------------------
  // Sequential
  //--------------------------------------------------------------------------
  always_ff @(posedge clk or negedge rst_n) begin
    if (!rst_n) begin
      state <= SM_IDLE;
    end else begin
      state <= next_state;
    end
  end

  //--------------------------------------------------------------------------
  // Next-state logic
  //--------------------------------------------------------------------------
  always_comb begin
    next_state = state;

    case (state)
      SM_IDLE: begin
        if (start)
          next_state = SM_LOAD_MAX;
      end

      SM_LOAD_MAX: begin
        if (max_done)
          next_state = SM_SUB_EXP;
      end

      SM_SUB_EXP: begin
        if (acc_sum_valid)
          next_state = SM_RECIPROCAL;
      end

      SM_RECIPROCAL: begin
        if (recip_done)
          next_state = SM_NORMALIZE;
      end

      SM_NORMALIZE: begin
        if (norm_last)
          next_state = SM_DONE;
      end

      SM_DONE: begin
        next_state = SM_IDLE;
      end

      default: next_state = SM_IDLE;
    endcase
  end

  //--------------------------------------------------------------------------
  // Output logic (Moore machine — outputs from state only)
  //--------------------------------------------------------------------------
  always_comb begin
    // Defaults
    max_start    = 1'b0;
    sub_start    = 1'b0;
    acc_start    = 1'b0;
    recip_start  = 1'b0;
    norm_start   = 1'b0;
    in_ready     = 1'b0;
    busy         = 1'b0;
    done         = 1'b0;

    case (state)
      SM_IDLE: begin
        in_ready  = 1'b0;
        if (start) begin
          max_start = 1'b1;
        end
      end

      SM_LOAD_MAX: begin
        busy     = 1'b1;
        in_ready = 1'b1; // Accept streaming input data
        if (max_done) begin
          sub_start = 1'b1;
          acc_start = 1'b1;
        end
      end

      SM_SUB_EXP: begin
        busy = 1'b1;
        if (acc_sum_valid) begin
          recip_start = 1'b1;
        end
      end

      SM_RECIPROCAL: begin
        busy = 1'b1;
        if (recip_done) begin
          norm_start = 1'b1;
        end
      end

      SM_NORMALIZE: begin
        busy = 1'b1;
      end

      SM_DONE: begin
        done = 1'b1;
      end

      default: ;
    endcase
  end

endmodule
