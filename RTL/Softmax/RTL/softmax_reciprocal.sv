//============================================================================
// Module:  softmax_reciprocal.sv
// Project: BERT Self-Attention Softmax Hardware Accelerator
// Description: Newton-Raphson reciprocal (1/a) with input normalization.
//
//              1. Normalize: shift a right until integer part = 1 
//                 (a_norm in [1.0, 2.0)). Record shift amount S.
//              2. LUT: 16-entry guess indexed by top 4 frac bits of a_norm
//              3. NR iteration (2x): y_{n+1} = y_n * (2 - a_norm * y_n)
//              4. De-normalize: result = y_final >> S
//
//              Input:  Q8.24 unsigned (32-bit)
//              Output: Q8.24 unsigned (32-bit)
//              Latency: 10 cycles (IDLE→NORM→LUT_LOAD→[MUL→SUB→UPD]×2→SHIFT→DONE)
//============================================================================

`timescale 1ns/1ps

module softmax_reciprocal
  import softmax_pkg::*;
#(
  parameter int W         = NR_W,
  parameter int Q         = NR_FRAC,
  parameter int LUT_BITS  = NR_LUT_BITS,
  parameter int ITER      = NR_ITER
) (
  input  logic                  clk,
  input  logic                  rst_n,
  input  logic                  in_valid,
  input  logic [W-1:0]          in_data,
  output logic                  out_valid,
  output logic [W-1:0]          out_data
);

  localparam int DW2 = 2 * W;
  localparam [W-1:0] TWO_Q = 32'd2 << Q;
  localparam int INT_BITS = W - Q;         // 8
  localparam int LUT_SIZE = 1 << LUT_BITS; // 16

  // LUT
  reg [W-1:0] init_lut [0:LUT_SIZE-1];
  initial begin : init_lut_block
    integer ii;
    for (ii = 0; ii < LUT_SIZE; ii = ii + 1)
      init_lut[ii] = {W{1'b0}};
    $readmemh("nr_init.hex", init_lut);
  end

  // FSM
  typedef enum logic [3:0] {
    ST_IDLE,
    ST_NORM,        // Normalize input + compute shift
    ST_LOAD_LUT,    // Load LUT guess using normalized value
    ST_MUL_AY,      // Compute a_norm * y_n
    ST_SUB,         // 2 - (a_norm*y_n >> Q)
    ST_UPDATE,      // y = y * (2-a*y) >> Q
    ST_SHIFT,       // De-normalize: result >> shift_amt
    ST_DONE
  } state_t;

  state_t state, next_state;

  // Registers
  reg [W-1:0]    a_norm;
  reg [W-1:0]    y_reg;
  reg [DW2-1:0]  mul_ay_reg;
  reg [W-1:0]    sub_reg;
  integer        iter_cnt;
  reg [3:0]      shift_amt;  // Max 7 for 8-bit integer part

  // MSB detection: find highest set bit in integer part
  // Compute shift and normalized value
  reg [INT_BITS-1:0] int_part_raw;
  reg [3:0]          msb_found;

  always_comb begin
    int_part_raw = in_data[W-1:Q];
    msb_found = 4'd0;
    if (int_part_raw[7])      msb_found = 4'd7;
    else if (int_part_raw[6]) msb_found = 4'd6;
    else if (int_part_raw[5]) msb_found = 4'd5;
    else if (int_part_raw[4]) msb_found = 4'd4;
    else if (int_part_raw[3]) msb_found = 4'd3;
    else if (int_part_raw[2]) msb_found = 4'd2;
    else if (int_part_raw[1]) msb_found = 4'd1;
    else                      msb_found = 4'd0;
  end

  // 64-bit multiplies (combinational, uses registered values)
  reg [DW2-1:0] an_ext, y_ext, ay_prod;
  always_comb begin
    an_ext  = {{W{1'b0}}, a_norm};
    y_ext   = {{W{1'b0}}, y_reg};
    ay_prod = an_ext * y_ext;
  end

  reg [DW2-1:0] sub_ext, ysub_prod;
  always_comb begin
    sub_ext   = {{W{1'b0}}, sub_reg};
    ysub_prod = y_ext * sub_ext;
  end

  // LUT index from normalized value: bits [Q-1:Q-LUT_BITS] = [23:20]
  // These are the top 4 fractional bits of a_norm (which has integer part = 1)
  wire [LUT_BITS-1:0] lut_idx = a_norm[Q-1 -: LUT_BITS];

  // FSM
  always_ff @(posedge clk or negedge rst_n) begin
    if (!rst_n) begin
      state       <= ST_IDLE;
      a_norm      <= {W{1'b0}};
      y_reg       <= {W{1'b0}};
      mul_ay_reg  <= {DW2{1'b0}};
      sub_reg     <= {W{1'b0}};
      iter_cnt    <= 0;
      shift_amt   <= 4'd0;
      out_valid   <= 1'b0;
      out_data    <= {W{1'b0}};
    end else begin
      state     <= next_state;
      out_valid <= 1'b0;

      case (state)
        ST_IDLE: begin
          if (in_valid) begin
            // Normalize: shift right by msb_found
            if (in_data[W-1:Q] == {INT_BITS{1'b0}}) begin
              // Input < 1.0: don't normalize, shift_amt = 0
              a_norm    <= in_data;
              shift_amt <= 4'd0;
            end else begin
              a_norm    <= in_data >> msb_found;
              shift_amt <= msb_found;
            end
          end
        end

        ST_NORM: begin
          // a_norm is now registered with integer part = 1
          // Nothing extra here, just a pipeline stage
        end

        ST_LOAD_LUT: begin
          // Load initial guess from LUT using normalized fractional bits
          y_reg    <= init_lut[lut_idx];
          iter_cnt <= 0;
        end

        ST_MUL_AY: begin
          mul_ay_reg <= ay_prod;
        end

        ST_SUB: begin
          sub_reg <= TWO_Q - mul_ay_reg[Q +: W];
        end

        ST_UPDATE: begin
          y_reg <= ysub_prod[Q +: W];
          if (iter_cnt < ITER - 1)
            iter_cnt <= iter_cnt + 1;
        end

        ST_SHIFT: begin
          out_data  <= y_reg >> shift_amt;
          out_valid <= 1'b1;
        end

        ST_DONE: begin
          // Result already output
        end

        default: ;
      endcase
    end
  end

  // Next-state logic
  always_comb begin
    next_state = state;
    case (state)
      ST_IDLE:     next_state = in_valid ? ST_NORM : ST_IDLE;
      ST_NORM:     next_state = ST_LOAD_LUT;
      ST_LOAD_LUT: next_state = ST_MUL_AY;
      ST_MUL_AY:   next_state = ST_SUB;
      ST_SUB:      next_state = ST_UPDATE;
      ST_UPDATE:   next_state = (iter_cnt < ITER - 1) ? ST_MUL_AY : ST_SHIFT;
      ST_SHIFT:    next_state = ST_DONE;
      ST_DONE:     next_state = ST_IDLE;
      default:     next_state = ST_IDLE;
    endcase
  end

endmodule
