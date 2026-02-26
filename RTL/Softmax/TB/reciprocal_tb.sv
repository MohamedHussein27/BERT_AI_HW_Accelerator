//============================================================================
// Testbench: reciprocal_tb.sv
// Project:   BERT Softmax Hardware Accelerator
// Description: Unit testbench for the Newton-Raphson reciprocal module.
//              Drives test inputs, compares against expected values.
//============================================================================

`timescale 1ns/1ps

module reciprocal_tb;

  import softmax_pkg::*;

  //--------------------------------------------------------------------------
  // Parameters
  //--------------------------------------------------------------------------
  parameter int NUM_TESTS = 50;
  parameter int CLK_PERIOD = 10;

  //--------------------------------------------------------------------------
  // Signals
  //--------------------------------------------------------------------------
  logic                  clk;
  logic                  rst_n;
  logic                  in_valid;
  logic [ACC_W-1:0]      in_data;
  logic                  out_valid;
  logic [ACC_W-1:0]      out_data;

  //--------------------------------------------------------------------------
  // DUT
  //--------------------------------------------------------------------------
  softmax_reciprocal #(
    .W         (ACC_W),
    .Q         (FRAC_ACC),
    .LUT_BITS  (NR_LUT_BITS),
    .ITER      (NR_ITER)
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
  logic [ACC_W-1:0] test_inputs   [0:NUM_TESTS-1];
  logic [ACC_W-1:0] test_expected [0:NUM_TESTS-1];

  initial begin
    $readmemh("recip_test_inputs.hex", test_inputs);
    $readmemh("recip_test_expected.hex", test_expected);
  end

  //--------------------------------------------------------------------------
  // Test process
  //--------------------------------------------------------------------------
  integer i;
  integer out_idx;
  integer pass_cnt;
  integer fail_cnt;
  real    max_rel_error;
  real    abs_err;
  real    rel_err;
  real    hw_float;
  real    exp_float;

  initial begin
    $display("============================================================");
    $display("  Newton-Raphson Reciprocal Unit Testbench");
    $display("============================================================");

    rst_n    = 0;
    in_valid = 0;
    in_data  = 0;
    out_idx  = 0;
    pass_cnt = 0;
    fail_cnt = 0;
    max_rel_error = 0.0;

    repeat(5) @(posedge clk);
    rst_n = 1;
    repeat(2) @(posedge clk);

    // Drive test vectors sequentially (each takes multiple cycles)
    for (i = 0; i < NUM_TESTS; i = i + 1) begin
      @(posedge clk);
      in_valid <= 1;
      in_data  <= test_inputs[i];
      @(posedge clk);
      in_valid <= 0;

      // Wait for result
      wait(out_valid);
      @(posedge clk);

      // Process result in output capture block
      repeat(3) @(posedge clk);
    end

    // Final wait
    repeat(20) @(posedge clk);

    $display("");
    $display("Results:");
    $display("  PASS: %0d / %0d", pass_cnt, out_idx);
    $display("  FAIL: %0d / %0d", fail_cnt, out_idx);
    $display("  Max Relative Error: %f%%", max_rel_error * 100.0);
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
      hw_float  = real'(out_data) / real'(1 << FRAC_ACC);
      exp_float = real'(test_expected[out_idx]) / real'(1 << FRAC_ACC);

      if (exp_float > 0.0) begin
        abs_err = (hw_float > exp_float) ? (hw_float - exp_float) : (exp_float - hw_float);
        rel_err = abs_err / exp_float;
      end else begin
        abs_err = 0.0;
        rel_err = 0.0;
      end

      if (rel_err > max_rel_error)
        max_rel_error = rel_err;

      // Allow 5% relative error tolerance (NR with 2 iterations)
      if (rel_err <= 0.05) begin
        pass_cnt = pass_cnt + 1;
      end else begin
        fail_cnt = fail_cnt + 1;
        $display("  FAIL[%0d]: input=%08h, got=%08h (%.6f), exp=%08h (%.6f), rel_err=%.4f%%",
                 out_idx, test_inputs[out_idx],
                 out_data, hw_float,
                 test_expected[out_idx], exp_float,
                 rel_err * 100.0);
      end

      out_idx = out_idx + 1;
    end
  end

endmodule
