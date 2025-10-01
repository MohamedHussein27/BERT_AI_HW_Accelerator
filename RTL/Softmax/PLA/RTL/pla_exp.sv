`timescale 1ns/1ps

module pla_exp_pipelined #(
  parameter int Q       = 26,            // fractional bits (Q5.26)
  parameter int W       = 32,            // data width (bits)
  parameter int H_SHIFT = 26,            // h = 2^H_SHIFT (in Q units)
  parameter int NSEG    = 32             // number of segments (ROM size)
) (
  input  logic                    clk,
  input  logic                    rst_n,     // async active-low reset
  input  logic                    start,     // 1-cycle start pulse
  input  logic signed  [W-1:0]    x_in_q,    // input x in Q format (signed)
  output logic                    busy,
  output logic                    done,
  output logic signed  [W-1:0]    y_exp_q    // exp(x) in Q format
);

  // domain
  localparam signed [W-1:0] XMIN = -16 << Q;
  localparam signed [W-1:0] XMAX =  16 << Q;

  // ROMs (signed)
  logic signed [W-1:0] w_mem [0:NSEG-1];
  logic signed [W-1:0] b_mem [0:NSEG-1];

  // zero init (simulation convenience)
  initial begin : init_roms
    integer ii;
    for (ii = 0; ii < NSEG; ii = ii + 1) begin
      w_mem[ii] = '0;
      b_mem[ii] = '0;
    end
  end

  // pipeline registers (signed where applicable)
  logic signed [W-1:0] x_clamped;      // combinational clamp
  logic signed [W-1:0] x_stage;        // stage 0 -> multiply
  logic signed [W-1:0] w_stage;        // registered slope
  logic signed [W-1:0] b_stage;        // registered intercept
  logic signed [63:0]  mul64_stage;    // registered 64-bit product
  logic signed [63:0]  scaled64_stage; // scaled 64-bit (product >>> Q)
  logic signed [W-1:0] add_stage;      // truncated result (registered)
  logic [15:0]         idx;            // index computed combinationally

  // Helper 64-bit b and sum (moved to module scope for simulators)
  logic signed [63:0]  b64;
  logic signed [63:0]  sum64;

  // FSM states
  typedef enum logic [2:0] {S_IDLE, S_CAPTURE, S_MUL, S_SCALE_ADD, S_DONE} state_t;
  state_t state, next_state;

  // ------- clamp x (combinational)
  always_comb begin
    if ($signed(x_in_q) < $signed(XMIN)) x_clamped = XMIN;
    else if ($signed(x_in_q) > $signed(XMAX)) x_clamped = XMAX;
    else x_clamped = x_in_q;
  end

  // ------- compute index (combinational) - H_SHIFT assumed Q for h=1.0
  always_comb begin
    logic signed [W-1:0] delta;
    delta = $signed(x_clamped) - $signed(XMIN);
    if (delta <= 0)
      idx = 0;
    else begin
      idx = $unsigned(delta) >>> H_SHIFT;
      if (idx >= NSEG) idx = NSEG - 1;
    end
  end

  // ------- sequential pipeline (registers & FSM)
  always_ff @(posedge clk or negedge rst_n) begin
    if (!rst_n) begin
      state         <= S_IDLE;
      busy          <= 1'b0;
      done          <= 1'b0;
      x_stage       <= '0;
      w_stage       <= '0;
      b_stage       <= '0;
      mul64_stage   <= 64'sd0;
      scaled64_stage<= 64'sd0;
      add_stage     <= '0;
      y_exp_q       <= '0;
      b64           <= 64'sd0;
      sum64         <= 64'sd0;
    end else begin
      state <= next_state;
      done  <= 1'b0; // default

      case (state)
        S_IDLE: begin
          busy <= 1'b0;
          if (start) begin
            // capture ROM entries and clamped input into registers
            w_stage <= w_mem[idx];
            b_stage <= b_mem[idx];
            x_stage <= x_clamped;
            busy    <= 1'b1;
          end
        end

        S_CAPTURE: begin
          // pipeline stage; nothing else required
        end

        S_MUL: begin
          // signed multiplication 32x32->64
          mul64_stage <= $signed(w_stage) * $signed(x_stage);
        end

        S_SCALE_ADD: begin
          // scale (arithmetic shift)
          scaled64_stage <= $signed(mul64_stage) >>> Q;

          // sign-extend b_stage to 64 bits (explicit)
          b64 <= { { (64-W){ b_stage[W-1] } }, b_stage };

          // add in 64-bit signed domain
          sum64 <= $signed(scaled64_stage) + $signed(b64);

          // truncate to W bits (lower W bits)
          add_stage <= sum64[W-1:0];
        end

        S_DONE: begin
          y_exp_q <= add_stage;
          done    <= 1'b1;
          busy    <= 1'b0;
        end

        default: begin end
      endcase
    end
  end

  // ------- next-state logic (pipeline progression)
  always_comb begin
    next_state = state;
    case (state)
      S_IDLE:      if (start) next_state = S_CAPTURE;
                   else next_state = S_IDLE;
      S_CAPTURE:   next_state = S_MUL;
      S_MUL:       next_state = S_SCALE_ADD;
      S_SCALE_ADD: next_state = S_DONE;
      S_DONE:      next_state = S_IDLE;
      default:     next_state = S_IDLE;
    endcase
  end

endmodule
