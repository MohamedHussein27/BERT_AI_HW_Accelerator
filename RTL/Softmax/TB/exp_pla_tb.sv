//============================================================================
// Testbench: exp_pla_tb.sv
// Project:   BERT Softmax Hardware Accelerator
// Description: Unit testbench for the PLA exponential module.
//              Loads ROM coefficients, drives test vectors from hex file,
//              compares DUT output against expected values.
//============================================================================

`timescale 1ns/1ps

module exp_pla_tb;

  import softmax_pkg::*;

  //--------------------------------------------------------------------------
  // Parameters
  //--------------------------------------------------------------------------
  parameter int NUM_TESTS = 200;
  parameter int CLK_PERIOD = 10;  // 100 MHz

  //--------------------------------------------------------------------------
  // Signals
  //--------------------------------------------------------------------------
  logic                    clk;
  logic                    rst_n;
  logic                    in_valid;
  logic signed [DATA_W-1:0] in_data;
  logic                    out_valid;
  logic [EXP_W-1:0]        out_data;

  //--------------------------------------------------------------------------
  // DUT
  //--------------------------------------------------------------------------
  softmax_exp_pla #(
    .D_W       (DATA_W),
    .FRAC_I    (FRAC_IN),
    .O_W       (EXP_W),
    .FRAC_O    (FRAC_EXP),
    .NSEG      (PLA_NSEG),
    .COEFF_W   (PLA_COEFF_W),
    .COEFF_F   (PLA_COEFF_F),
    .H_SHIFT   (PLA_H_SHIFT)
  ) dut (
    .clk       (clk),
    .rst_n     (rst_n),
    .in_valid  (in_valid),
    .in_data   (in_data),
    .out_valid (out_valid),
    .out_data  (out_data)
  );

  //--------------------------------------------------------------------------
  // Clock generation
  //--------------------------------------------------------------------------
  initial clk = 0;
  always #(CLK_PERIOD/2) clk = ~clk;

  //--------------------------------------------------------------------------
  // Test vectors
  //--------------------------------------------------------------------------
  logic [DATA_W-1:0] test_inputs   [0:NUM_TESTS-1];
  logic [EXP_W-1:0]  test_expected [0:NUM_TESTS-1];

  initial begin
    $readmemh("exp_test_inputs.hex", test_inputs);
    $readmemh("exp_test_expected.hex", test_expected);
  end

  //--------------------------------------------------------------------------
  // Test process
  //--------------------------------------------------------------------------
  integer i;
  integer out_idx;
  integer pass_cnt;
  integer fail_cnt;
  real    max_abs_error;
  real    sum_sq_error;
  real    abs_err;
  real    hw_val;
  real    exp_val;

  initial begin
    $display("============================================================");
    $display("  PLA Exponential Unit Testbench");
    $display("============================================================");

    // Reset
    rst_n    = 0;
    in_valid = 0;
    in_data  = 0;
    out_idx  = 0;
    pass_cnt = 0;
    fail_cnt = 0;
    max_abs_error = 0.0;
    sum_sq_error  = 0.0;

    repeat(5) @(posedge clk);
    rst_n = 1;
    repeat(2) @(posedge clk);

    // Drive test vectors
    for (i = 0; i < NUM_TESTS; i = i + 1) begin
      @(posedge clk);
      in_valid <= 1;
      in_data  <= test_inputs[i];
    end
    @(posedge clk);
    in_valid <= 0;

    // Wait for pipeline to flush
    repeat(10) @(posedge clk);

    $display("");
    $display("Results:");
    $display("  PASS: %0d / %0d", pass_cnt, NUM_TESTS);
    $display("  FAIL: %0d / %0d", fail_cnt, NUM_TESTS);
    $display("  Max Abs Error (Q1.15 units): %0d", $rtoi(max_abs_error));
    $display("  MSE (Q1.15 units):           %f", sum_sq_error / NUM_TESTS);
    $display("============================================================");

    if (fail_cnt == 0)
      $display("  >>> ALL TESTS PASSED <<<");
    else
      $display("  >>> SOME TESTS FAILED <<<");

    $finish;
  end

  // Capture outputs
  always @(posedge clk) begin
    if (out_valid && out_idx < NUM_TESTS) begin
      abs_err = (out_data > test_expected[out_idx]) ?
                (out_data - test_expected[out_idx]) :
                (test_expected[out_idx] - out_data);

      sum_sq_error = sum_sq_error + abs_err * abs_err;
      if (abs_err > max_abs_error)
        max_abs_error = abs_err;

      // Allow tolerance of 2 LSBs in Q1.15
      if (abs_err <= 2) begin
        pass_cnt = pass_cnt + 1;
      end else begin
        fail_cnt = fail_cnt + 1;
        $display("  FAIL[%0d]: input=%08h, got=%04h, expected=%04h, err=%0d",
                 out_idx, test_inputs[out_idx], out_data, test_expected[out_idx],
                 $rtoi(abs_err));
      end

      out_idx = out_idx + 1;
    end
  end

endmodule
