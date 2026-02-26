// nr_recip.v
// Newton-Raphson reciprocal (1/a) accelerator in Verilog-2001 style.
// - WIDTH-bit signed fixed-point with Q fractional bits (Q fractional bits).
// - Initial guess from small ROM of size 2^LUT_BITS (loaded via $readmemh in TB).
// - ITER iterations of NR: x_{n+1} = x_n * (2 - a * x_n)
// - Simple pipelined iterative design: values progress across cycles through pipeline regs.
// - All declarations at module scope; uses `always` (no SystemVerilog always_ff/always_comb).

`timescale 1ns/1ps
module nr_recip #
(
  parameter integer WIDTH    = 32,   // data width (bits)
  parameter integer Q        = 26,   // fractional bits (e.g. Q=26 => Q5.26)
  parameter integer LUT_BITS = 4,    // LUT index bits -> LUT size = 2^LUT_BITS
  parameter integer ITER     = 3     // number of Newton-Raphson iterations
)
(
  input  wire                        clk,
  input  wire                        rst_n,    // active-low async reset
  input  wire                        start,    // 1-cycle start pulse
  input  wire signed [WIDTH-1:0]     a_in_q,   // input a in Q format (signed, >0 expected)
  output reg                         busy,
  output reg                         done,     // 1-cycle pulse when y_out_q valid
  output reg signed [WIDTH-1:0]      y_out_q   // reciprocal 1/a in Q format
);

  // ---------------------------------------------------------------------------
  // Derived parameters / sizes
  // ---------------------------------------------------------------------------
  // Note: require LUT_BITS >= 1 for sensible behavior; if LUT_BITS==0 you'd adjust code.
  parameter integer LUT_SIZE  = (1 << LUT_BITS);
  parameter integer IDX_W     = (LUT_BITS == 0) ? 1 : LUT_BITS;
  parameter integer SHIFT_AMT = WIDTH - IDX_W;
  parameter integer MASK_IDX  = ( (1 << IDX_W) - 1 );

  // ---------------------------------------------------------------------------
  // ROM for initial guess
  // ---------------------------------------------------------------------------
  // Each word is WIDTH bits signed Q format. Load this in TB:
  //   $readmemh("nr_init.hex", uut.init_rom);
  reg signed [WIDTH-1:0] init_rom [0:LUT_SIZE-1];

  // create zero init so simulation doesn't see X when TB forgets to load
  integer zi;
  initial begin
    for (zi = 0; zi < LUT_SIZE; zi = zi + 1) begin
      init_rom[zi] = {WIDTH{1'b0}};
    end
  end

  // ---------------------------------------------------------------------------
  // Constants
  // ---------------------------------------------------------------------------
  // 2.0 in Q format (WIDTH bits signed)
  localparam signed [WIDTH-1:0] TWO_Q = (2 << Q);
  // widen 2.0 to 2*WIDTH bits (sign-extended)
  localparam signed [2*WIDTH-1:0] TWO_2W = { {WIDTH{TWO_Q[WIDTH-1]}}, TWO_Q };

  // ---------------------------------------------------------------------------
  // Module-scope registers / pipeline registers
  // ---------------------------------------------------------------------------
  reg signed [WIDTH-1:0]   a_reg;        // latched input a
  reg signed [WIDTH-1:0]   x_reg;        // current iterate x_n
  reg [IDX_W-1:0]          idx_reg;      // latched index to ROM
  integer                  iter_cnt;     // iteration counter
  reg                      running;      // internal busy flag

  // pipeline / datapath registers (2*WIDTH to hold products)
  reg signed [2*WIDTH-1:0] mul_ax_r;        // registered a * x (2W)
  reg signed [2*WIDTH-1:0] scaled_ax_r;     // registered (a*x) >>> Q (2W)
  reg signed [2*WIDTH-1:0] two_minus_ax_r;  // registered (2 - scaled_ax) (2W)
  reg signed [2*WIDTH-1:0] mult_tmp_r;      // registered x * (2 - ax) (2W)
  reg signed [WIDTH-1:0]   next_x_r;        // truncated next iterate (W)

  // combinational index extraction (top magnitude bits)
  wire [IDX_W-1:0] idx_comb;
  wire [WIDTH-1:0] unsigned_a;
  assign unsigned_a = a_in_q; // bit-copies; arithmetic on signed is done with $signed in regs
  assign idx_comb = (a_in_q <= 0) ? {IDX_W{1'b0}} :
                    ( (unsigned_a >> SHIFT_AMT) & MASK_IDX );

  // busy output mirrors running
  // busy is reg; we will update it in sequential block.

  // ---------------------------------------------------------------------------
  // Sequential pipeline & control
  // - Non-blocking assignments ensure the pipeline advances each clock:
  //   mul_ax_r <= a_reg * x_reg;               // computes a*x using current x_reg
  //   scaled_ax_r <= mul_ax_r >>> Q;           // uses previous mul_ax_r
  //   two_minus_ax_r <= TWO_2W - (mul_ax_r >>> Q);
  //   mult_tmp_r <= x_reg * two_minus_ax_r[WIDTH-1:0];
  //   next_x_r <= mult_tmp_r >>> Q;
  //   x_reg <= next_x_r;  etc.
  // ---------------------------------------------------------------------------
  integer i;
  always @(posedge clk or negedge rst_n) begin
    if (!rst_n) begin
      // reset
      a_reg          <= {WIDTH{1'b0}};
      x_reg          <= {WIDTH{1'b0}};
      idx_reg        <= {IDX_W{1'b0}};
      iter_cnt       <= 0;
      running        <= 1'b0;
      mul_ax_r       <= {2*WIDTH{1'b0}};
      scaled_ax_r    <= {2*WIDTH{1'b0}};
      two_minus_ax_r <= {2*WIDTH{1'b0}};
      mult_tmp_r     <= {2*WIDTH{1'b0}};
      next_x_r       <= {WIDTH{1'b0}};
      y_out_q        <= {WIDTH{1'b0}};
      done           <= 1'b0;
      busy           <= 1'b0;
    end else begin
      // default
      done <= 1'b0;
      // note: busy updated at end of cycle from running, but we'll also set active state inside
      if (start && !running) begin
        // latch inputs and initial guess
        a_reg    <= a_in_q;
        idx_reg  <= idx_comb;
        x_reg    <= init_rom[idx_comb]; // NOTE: using idx_comb (combinational) is fine here
        iter_cnt <= 0;
        running  <= 1'b1;
        // optionally clear pipeline regs
        mul_ax_r       <= {2*WIDTH{1'b0}};
        scaled_ax_r    <= {2*WIDTH{1'b0}};
        two_minus_ax_r <= {2*WIDTH{1'b0}};
        mult_tmp_r     <= {2*WIDTH{1'b0}};
        next_x_r       <= {WIDTH{1'b0}};
        y_out_q        <= {WIDTH{1'b0}};
        busy <= 1'b1;
      end else if (running) begin
        // Dataflow pipeline (non-blocking ensures previous values are used on RHS)
        // Stage A: start multiply a * x (using current a_reg and x_reg)
        mul_ax_r <= $signed(a_reg) * $signed(x_reg); // 2*WIDTH product

        // Stage B: use previous mul_ax_r (from prior cycle) to produce scaled_ax etc.
        // scaled_ax_r = mul_ax_r >>> Q (arithmetic shift)
        scaled_ax_r <= mul_ax_r >>> Q;

        // two_minus_ax_r = 2 - scaled_ax_r
        two_minus_ax_r <= TWO_2W - (mul_ax_r >>> Q);

        // Multiply x_reg by lower WIDTH bits of two_minus to produce next product
        // (we use two_minus_ax_r[WIDTH-1:0] which is the lower WIDTH bits)
        mult_tmp_r <= $signed(x_reg) * $signed(two_minus_ax_r[WIDTH-1:0]);

        // next iterate (after shifting by Q)
        next_x_r <= mult_tmp_r >>> Q;

        // update iterate register with previously computed next value
        x_reg <= mult_tmp_r >>> Q;

        // iteration counter & termination
        if (iter_cnt < (ITER - 1)) begin
          iter_cnt <= iter_cnt + 1;
        end else begin
          // final iteration finished; produce output and stop
          iter_cnt <= 0;
          running  <= 1'b0;
          y_out_q  <= mult_tmp_r >>> Q;
          done     <= 1'b1;
          busy     <= 1'b0;
        end
      end else begin
        // not running; ensure busy low
        busy <= 1'b0;
      end
    end
  end

  // end module
endmodule
