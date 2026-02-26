//============================================================================
// Testbench: accumulator_tb.sv
// Project:   BERT Softmax Hardware Accelerator
// Description: Unit testbench for the softmax accumulator module.
//              Streams known Q1.15 exp values, verifies the accumulated sum.
//============================================================================

`timescale 1ns/1ps

module accumulator_tb;

  import softmax_pkg::*;

  //--------------------------------------------------------------------------
  // Parameters
  //--------------------------------------------------------------------------
  parameter int VEC_LEN    = 64;
  parameter int CLK_PERIOD = 10;

  //--------------------------------------------------------------------------
  // Signals
  //--------------------------------------------------------------------------
  logic                  clk;
  logic                  rst_n;
  logic                  start;
  logic [SEQ_IDX_W-1:0]  vec_len_cfg;
  logic                  in_valid;
  logic [EXP_W-1:0]      in_data;
  logic                  sum_valid;
  logic [ACC_W-1:0]      sum_out;
  logic [SEQ_IDX_W-1:0]  rd_addr;
  logic [EXP_W-1:0]      rd_data;

  //--------------------------------------------------------------------------
  // DUT
  //--------------------------------------------------------------------------
  softmax_accumulator #(
    .I_W     (EXP_W),
    .FRAC_I  (FRAC_EXP),
    .O_W     (ACC_W),
    .FRAC_O  (FRAC_ACC),
    .MAX_LEN (SEQ_LEN_MAX),
    .IDX_W   (SEQ_IDX_W)
  ) dut (
    .clk         (clk),
    .rst_n       (rst_n),
    .start       (start),
    .vec_len_cfg (vec_len_cfg),
    .in_valid    (in_valid),
    .in_data     (in_data),
    .sum_valid   (sum_valid),
    .sum_out     (sum_out),
    .rd_addr     (rd_addr),
    .rd_data     (rd_data)
  );

  //--------------------------------------------------------------------------
  // Clock generation
  //--------------------------------------------------------------------------
  initial clk = 0;
  always #(CLK_PERIOD/2) clk = ~clk;

  //--------------------------------------------------------------------------
  // Test process
  //--------------------------------------------------------------------------
  integer i;
  real    expected_sum_f;
  real    actual_sum_f;
  real    abs_err;
  logic [ACC_W-1:0] expected_sum_q;
  integer pass_cnt;
  integer test_num;

  // Test data: uniform Q1.15 values
  // Each value = 0.5 in Q1.15 = 16384
  localparam logic [EXP_W-1:0] TEST_VAL = 16'd16384; // 0.5 in Q1.15

  initial begin
    $display("============================================================");
    $display("  Softmax Accumulator Unit Testbench");
    $display("============================================================");

    rst_n       = 0;
    start       = 0;
    in_valid    = 0;
    in_data     = 0;
    vec_len_cfg = VEC_LEN;
    rd_addr     = 0;
    pass_cnt    = 0;
    test_num    = 0;

    repeat(5) @(posedge clk);
    rst_n = 1;
    repeat(2) @(posedge clk);

    // ======================================================================
    // Test 1: All values = 0.5, vector length = 64
    // Expected sum = 64 * 0.5 = 32.0 in Q8.24 = 32 * 2^24 = 536870912
    // ======================================================================
    test_num = 1;
    $display("\nTest %0d: All values = 0.5, VEC_LEN = %0d", test_num, VEC_LEN);
    expected_sum_f = VEC_LEN * 0.5;
    expected_sum_q = expected_sum_f * (1 << FRAC_ACC);

    // Start accumulator
    @(posedge clk);
    start <= 1;
    @(posedge clk);
    start <= 0;

    // Stream values
    for (i = 0; i < VEC_LEN; i = i + 1) begin
      @(posedge clk);
      in_valid <= 1;
      in_data  <= TEST_VAL;
    end
    @(posedge clk);
    in_valid <= 0;

    // Wait for sum
    wait(sum_valid);
    @(posedge clk);

    actual_sum_f = real'(sum_out) / real'(1 << FRAC_ACC);
    abs_err = (actual_sum_f > expected_sum_f) ?
              (actual_sum_f - expected_sum_f) :
              (expected_sum_f - actual_sum_f);

    $display("  Expected sum: %.4f (0x%08h)", expected_sum_f, expected_sum_q);
    $display("  Actual sum:   %.4f (0x%08h)", actual_sum_f, sum_out);
    $display("  Abs error:    %.6f", abs_err);

    if (abs_err < 0.001) begin
      $display("  >>> PASS <<<");
      pass_cnt = pass_cnt + 1;
    end else begin
      $display("  >>> FAIL <<<");
    end

    // Verify SRAM contents
    repeat(2) @(posedge clk);
    $display("  Verifying SRAM contents...");
    for (i = 0; i < 4; i = i + 1) begin
      rd_addr = i;
      #1; // Allow combinational read
      if (rd_data !== TEST_VAL) begin
        $display("  SRAM[%0d] MISMATCH: got=%04h, expected=%04h", i, rd_data, TEST_VAL);
      end
    end
    $display("  SRAM spot-check complete.");

    // ======================================================================
    // Test 2: All values = 1.0, vector length = 8
    // Expected sum = 8.0 in Q8.24
    // ======================================================================
    test_num = 2;
    $display("\nTest %0d: All values = 1.0, VEC_LEN = 8", test_num);
    expected_sum_f = 8.0;
    vec_len_cfg = 8;

    repeat(3) @(posedge clk);
    @(posedge clk);
    start <= 1;
    @(posedge clk);
    start <= 0;

    for (i = 0; i < 8; i = i + 1) begin
      @(posedge clk);
      in_valid <= 1;
      in_data  <= 16'h8000; // 1.0 in Q1.15 = 32768
    end
    @(posedge clk);
    in_valid <= 0;

    wait(sum_valid);
    @(posedge clk);

    actual_sum_f = real'(sum_out) / real'(1 << FRAC_ACC);
    abs_err = (actual_sum_f > expected_sum_f) ?
              (actual_sum_f - expected_sum_f) :
              (expected_sum_f - actual_sum_f);

    $display("  Expected sum: %.4f", expected_sum_f);
    $display("  Actual sum:   %.4f (0x%08h)", actual_sum_f, sum_out);
    $display("  Abs error:    %.6f", abs_err);

    if (abs_err < 0.001) begin
      $display("  >>> PASS <<<");
      pass_cnt = pass_cnt + 1;
    end else begin
      $display("  >>> FAIL <<<");
    end

    // ======================================================================
    // Summary
    // ======================================================================
    repeat(5) @(posedge clk);
    $display("");
    $display("============================================================");
    $display("  Results: %0d / %0d tests passed", pass_cnt, test_num);
    $display("============================================================");
    $finish;
  end

endmodule
