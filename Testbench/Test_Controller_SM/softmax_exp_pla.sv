//============================================================================
// Module:  softmax_exp_pla.sv
// Project: BERT Self-Attention Softmax Hardware Accelerator
// Description: Piecewise Linear Approximation (PLA) for exp(x) where x <= 0.
//              Streaming design: accepts one Q5.26 input per clock,
//              produces one Q1.15 unsigned output after a 3-cycle pipeline.
//
//              Approximation: exp(x) ≈ slope[idx] * (x - x0) + exp(x0)
//              where x0 = xmin + idx * h (segment start)
//              Domain: [-16, 0], 32 segments, h = 0.5
//              Local offset (x - x0) is always in [0, h) = [0, 0.5)
//
//              Pipeline:
//                Stage 0: Clamp + index + ROM lookup + compute local offset
//                Stage 1: Multiply slope * x_local (registered)
//                Stage 2: Scale + add intercept + clamp (registered output)
//============================================================================

`timescale 1ns/1ps

module softmax_exp_pla
  import softmax_pkg::*;
#(
  parameter int D_W       = DATA_W,       // Input width (32, Q5.26)
  parameter int FRAC_I    = FRAC_IN,      // Input frac bits (26)
  parameter int O_W       = EXP_W,        // Output width (16, Q1.15)
  parameter int FRAC_O    = FRAC_EXP,     // Output frac bits (15)
  parameter int NSEG      = PLA_NSEG,     // Number of segments (32)
  parameter int COEFF_W   = PLA_COEFF_W,  // Coefficient width (16)
  parameter int COEFF_F   = PLA_COEFF_F,  // Coefficient frac bits (15)
  parameter int H_SHIFT   = PLA_H_SHIFT   // Shift for segment width (25)
) (
  input  logic                    clk,
  input  logic                    rst_n,

  // Streaming input
  input  logic                    in_valid,
  input  logic signed [D_W-1:0]  in_data,   // Q5.26 signed, expected <= 0

  // Streaming output
  output logic                    out_valid,
  output logic [O_W-1:0]          out_data   // Q1.15 unsigned
);

  //--------------------------------------------------------------------------
  // Constants
  //--------------------------------------------------------------------------
  localparam int IDX_W = $clog2(NSEG);

  // Domain in Q5.26
  localparam signed [D_W-1:0] XMIN = -32'sd1073741824; // -16 << 26
  localparam signed [D_W-1:0] XMAX = 32'sd0;            // 0 in Q5.26

  //--------------------------------------------------------------------------
  // ROM for slopes (w) and intercepts (b) — Q1.15 unsigned stored as 16-bit
  // w[i] = (exp(x1) - exp(x0)) / h   where x0 = xmin + i*h, x1 = x0 + h
  // b[i] = exp(x0)                   (value at segment start)
  //--------------------------------------------------------------------------
  reg [COEFF_W-1:0] w_rom [0:NSEG-1];
  reg [COEFF_W-1:0] b_rom [0:NSEG-1];

  initial begin : init_roms
    integer ii;
    for (ii = 0; ii < NSEG; ii = ii + 1) begin
      w_rom[ii] = {COEFF_W{1'b0}};
      b_rom[ii] = {COEFF_W{1'b0}};
    end
    $readmemh("pla_slopes.hex", w_rom);
    $readmemh("pla_intercepts.hex", b_rom);
  end

  //--------------------------------------------------------------------------
  // Stage 0: Clamp + Index + ROM lookup + Local offset
  //--------------------------------------------------------------------------
  logic signed [D_W-1:0]  s0_x_clamped;
  logic [IDX_W-1:0]       s0_idx;
  logic [D_W-1:0]         s0_delta;  // x_clamped - XMIN (always positive)
  logic [D_W-1:0]         s0_x_local; // (x - x0) in Q5.26 unsigned, range [0, h)

  // Clamp input to [XMIN, XMAX]
  always_comb begin
    if ($signed(in_data) < $signed(XMIN))
      s0_x_clamped = XMIN;
    else if ($signed(in_data) > $signed(XMAX))
      s0_x_clamped = XMAX;
    else
      s0_x_clamped = in_data;
  end

  // Compute segment index and local offset
  // Note: extra bit needed since NSEG=32 doesn't fit in IDX_W=5 bits
  localparam [IDX_W:0] NSEG_WIDE = NSEG;            // 6-bit: 100000
  localparam [IDX_W-1:0] MAX_IDX = NSEG[IDX_W:0] - 1; // 31 = 11111

  always_comb begin
    s0_delta = s0_x_clamped - XMIN; // Always positive since x >= XMIN

    // Extract raw segment index from delta[H_SHIFT +: (IDX_W+1)] using wider range
    // to detect overflow (idx >= NSEG)
    if (s0_delta[H_SHIFT + IDX_W]) // Bit 30 set => idx >= 32, clamp
      s0_idx = MAX_IDX;
    else
      s0_idx = s0_delta[H_SHIFT +: IDX_W];

    // Local offset: lower H_SHIFT bits of delta
    s0_x_local = {{(D_W-H_SHIFT){1'b0}}, s0_delta[H_SHIFT-1:0]};
  end

  // Pipeline register: Stage 0 -> Stage 1
  reg                    s1_valid;
  reg [D_W-1:0]         s1_x_local; // Local offset, unsigned, [0, h) in Q5.26
  reg [COEFF_W-1:0]     s1_w;
  reg [COEFF_W-1:0]     s1_b;

  always @(posedge clk or negedge rst_n) begin
    if (!rst_n) begin
      s1_valid   <= 1'b0;
      s1_x_local <= {D_W{1'b0}};
      s1_w       <= {COEFF_W{1'b0}};
      s1_b       <= {COEFF_W{1'b0}};
    end else begin
      s1_valid <= in_valid;
      if (in_valid) begin
        s1_x_local <= s0_x_local;
        s1_w       <= w_rom[s0_idx];
        s1_b       <= b_rom[s0_idx];
      end
    end
  end

  //--------------------------------------------------------------------------
  // Stage 1: Multiply  ->  product = w * x_local
  //--------------------------------------------------------------------------
  // w is slope: unsigned Q1.15, 16-bit
  // x_local is local offset: unsigned Q5.26, 32-bit (but always < 0.5)
  //
  // product = w * x_local : both unsigned
  // Product format: Q(1+5).(15+26) = Q6.41, 48-bit UNSIGNED
  // To get Q1.15 portion: shift right by (41 - 15) = 26 = FRAC_I
  //
  // Since both operands are unsigned and small (w < 2.0, x_local < 0.5),
  // the product is at most ~1.0, which fits easily in Q1.15.

  localparam int PROD_W = COEFF_W + D_W; // 48 bits

  // CRITICAL: Verilog multiply result width = max(operand_widths).
  // For 16b * 32b, result is only 32 bits unless we extend operands.
  // Explicitly zero-extend both to 48 bits before multiply.
  reg [PROD_W-1:0] s1_w_ext;      // w zero-extended to 48 bits
  reg [PROD_W-1:0] s1_xl_ext;     // x_local zero-extended to 48 bits
  reg [PROD_W-1:0] s1_product;    // 48-bit unsigned product

  always_comb begin
    s1_w_ext    = {{(PROD_W-COEFF_W){1'b0}}, s1_w};       // 16 -> 48
    s1_xl_ext   = {{(PROD_W-D_W){1'b0}}, s1_x_local};     // 32 -> 48
    s1_product  = s1_w_ext * s1_xl_ext;                     // 48 x 48 -> 48
  end

  // Pipeline register: Stage 1 -> Stage 2
  reg                      s2_valid;
  reg [PROD_W-1:0]        s2_product;
  reg [COEFF_W-1:0]       s2_b;

  always @(posedge clk or negedge rst_n) begin
    if (!rst_n) begin
      s2_valid   <= 1'b0;
      s2_product <= {PROD_W{1'b0}};
      s2_b       <= {COEFF_W{1'b0}};
    end else begin
      s2_valid   <= s1_valid;
      s2_product <= s1_product;
      s2_b       <= s1_b;
    end
  end

  //--------------------------------------------------------------------------
  // Stage 2: Scale product + add intercept + output
  //--------------------------------------------------------------------------
  // product is Q6.41 (48-bit unsigned, but value < 1.0)
  // Shift right by FRAC_I (26) → Q6.15 unsigned
  // Then add b (Q1.15 unsigned) = exp(x0)
  // Result is exp(x) ≈ slope*(x-x0) + exp(x0), in Q1.15

  reg [COEFF_W:0]  s2_scaled; // 17 bits for the shifted product (max ~1.0)
  reg [COEFF_W:0]  s2_sum;    // 17 bits to hold potential carry

  always_comb begin
    // Shift right by FRAC_I (26 bits) to convert Q6.41 -> Q6.15
    s2_scaled = s2_product[FRAC_I +: (COEFF_W+1)]; // Extract 17 bits [42:26]
    // Add intercept (exp(x0))
    s2_sum = s2_scaled + {1'b0, s2_b};
  end

  // Output: clamp to [0, max_Q1.15] and register
  always @(posedge clk or negedge rst_n) begin
    if (!rst_n) begin
      out_valid <= 1'b0;
      out_data  <= {O_W{1'b0}};
    end else begin
      out_valid <= s2_valid;
      if (s2_valid) begin
        if (s2_sum[COEFF_W]) // Overflow bit set → saturate
          out_data <= {O_W{1'b1}};
        else
          out_data <= s2_sum[O_W-1:0];
      end else begin
        out_data <= {O_W{1'b0}};
      end
    end
  end

endmodule
