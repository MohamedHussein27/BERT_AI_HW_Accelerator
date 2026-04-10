//============================================================================
// Module:  softmax_pkg.sv
// Project: BERT Self-Attention Softmax Hardware Accelerator
// Description: Package containing all parameters, fixed-point types, and
//              helper functions for the Softmax pipeline.
//============================================================================

package softmax_pkg;

  //--------------------------------------------------------------------------
  // Global Parameters
  //--------------------------------------------------------------------------
  parameter int DATA_W      = 32;          // Input data width (Q5.26 signed)
  parameter int FRAC_IN     = 26;          // Input fractional bits
  parameter int INT_IN      = DATA_W - FRAC_IN - 1; // Input integer bits (5)

  parameter int EXP_W       = 16;          // Exp output width (Q1.15 unsigned)
  parameter int FRAC_EXP    = 15;          // Exp fractional bits

  parameter int ACC_W       = 32;          // Accumulator width (Q8.24 unsigned)
  parameter int FRAC_ACC    = 24;          // Accumulator fractional bits

  parameter int NORM_W      = 16;          // Normalized output width (Q1.15)
  parameter int FRAC_NORM   = 15;          // Normalized fractional bits

  parameter int SEQ_LEN     = 64;          // Default vector length (BERT head dim)
  parameter int SEQ_LEN_MAX = 512;         // Maximum supported sequence length
  parameter int SEQ_IDX_W   = $clog2(SEQ_LEN_MAX + 1); // Index width (must hold value SEQ_LEN_MAX)

  //--------------------------------------------------------------------------
  // PLA Exponential Parameters
  //--------------------------------------------------------------------------
  parameter int PLA_NSEG    = 32;          // Number of PLA segments
  parameter int PLA_IDX_W   = $clog2(PLA_NSEG); // Segment index width (5)

  // Domain for exp after max subtraction: [-16, 0]
  // In Q5.26: XMIN = -16 << 26, XMAX = 0
  // Segment width h = 16/32 = 0.5 => h_q = 0.5 * 2^26 = 2^25
  parameter int PLA_H_SHIFT = FRAC_IN - 1; // Right-shift amount for h=0.5 (25)

  // Slopes and intercepts stored as Q1.15 (16-bit) in ROM
  parameter int PLA_COEFF_W = 16;
  parameter int PLA_COEFF_F = 15;

  //--------------------------------------------------------------------------
  // Newton-Raphson Reciprocal Parameters
  //--------------------------------------------------------------------------
  parameter int NR_LUT_BITS = 4;           // LUT index width
  parameter int NR_LUT_SIZE = 1 << NR_LUT_BITS; // 16 entries
  parameter int NR_ITER     = 2;           // Number of NR iterations
  parameter int NR_W        = ACC_W;       // Working width (32b)
  parameter int NR_FRAC     = FRAC_ACC;    // Working fractional bits (24)

  //--------------------------------------------------------------------------
  // FSM States
  //--------------------------------------------------------------------------
  typedef enum logic [2:0] {
    SM_IDLE       = 3'b000,
    SM_LOAD_MAX   = 3'b001,
    SM_SUB_EXP    = 3'b010,
    SM_RECIPROCAL = 3'b011,
    SM_NORMALIZE  = 3'b100,
    SM_DONE       = 3'b101
  } sm_state_t;

  //--------------------------------------------------------------------------
  // Helper Functions
  //--------------------------------------------------------------------------

  // Convert real to Q5.26 signed 32-bit (for simulation only)
  function automatic logic [DATA_W-1:0] real_to_q526(input real val);
    longint temp;
    temp = longint'(val * real'(1 << FRAC_IN));
    return temp[DATA_W-1:0];
  endfunction

  // Convert Q1.15 unsigned 16-bit to real (for simulation only)
  function automatic real q115_to_real(input logic [EXP_W-1:0] val);
    return real'(val) / real'(1 << FRAC_EXP);
  endfunction

  // Convert Q8.24 unsigned 32-bit to real (for simulation only)
  function automatic real q824_to_real(input logic [ACC_W-1:0] val);
    return real'(val) / real'(1 << FRAC_ACC);
  endfunction

endpackage
