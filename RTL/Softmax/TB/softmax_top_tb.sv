//============================================================================
// Testbench: softmax_top_tb.sv
// Project:   BERT Softmax Hardware Accelerator
// Description: Integration testbench for bert_softmax top module.
//              Drives random test vectors through the full pipeline,
//              compares output against golden Python model hex files.
//              Reports MSE, max error, and per-vector pass/fail.
//============================================================================

`timescale 1ns/1ps

module softmax_top_tb;

  import softmax_pkg::*;

  //--------------------------------------------------------------------------
  // Parameters
  //--------------------------------------------------------------------------
  parameter int VEC_LEN    = 64;
  parameter int NUM_VECTORS = 10;
  parameter int TOTAL_ELEMS = VEC_LEN * NUM_VECTORS;
  parameter int CLK_PERIOD  = 10;  // 100 MHz
  parameter int TIMEOUT     = 500000; // Max simulation cycles

  //--------------------------------------------------------------------------
  // Signals
  //--------------------------------------------------------------------------
  logic                    clk;
  logic                    rst_n;
  logic                    start;
  logic [SEQ_IDX_W-1:0]    vec_len_cfg;
  logic                    in_valid;
  logic                    in_ready;
  logic signed [DATA_W-1:0] in_data;
  logic                    out_valid;
  logic [NORM_W-1:0]       out_data;
  logic                    out_last;
  logic                    busy;
  logic                    done;

  //--------------------------------------------------------------------------
  // DUT
  //--------------------------------------------------------------------------
  bert_softmax #(
    .VEC_LEN  (VEC_LEN),
    .D_W      (DATA_W),
    .O_W      (NORM_W),
    .MAX_LEN  (SEQ_LEN_MAX),
    .IDX_W    (SEQ_IDX_W)
  ) dut (
    .clk         (clk),
    .rst_n       (rst_n),
    .start       (start),
    .vec_len_cfg (vec_len_cfg),
    .in_valid    (in_valid),
    .in_ready    (in_ready),
    .in_data     (in_data),
    .out_valid   (out_valid),
    .out_data    (out_data),
    .out_last    (out_last),
    .busy        (busy),
    .done        (done)
  );

  //--------------------------------------------------------------------------
  // Clock generation
  //--------------------------------------------------------------------------
  initial clk = 0;
  always #(CLK_PERIOD/2) clk = ~clk;

  //--------------------------------------------------------------------------
  // Test vectors (loaded from hex files)
  //--------------------------------------------------------------------------
  logic [DATA_W-1:0] all_inputs   [0:TOTAL_ELEMS-1];
  logic [NORM_W-1:0] all_expected [0:TOTAL_ELEMS-1];

  initial begin
    $readmemh("input_vectors.hex", all_inputs);
    $readmemh("expected_outputs.hex", all_expected);
  end

  //--------------------------------------------------------------------------
  // Output capture
  //--------------------------------------------------------------------------
  integer out_idx;
  integer vec_out_cnt;
  integer vec_num;
  integer pass_cnt;
  integer fail_cnt;
  real    max_abs_error;
  real    sum_sq_error;
  real    total_elements;
  real    vec_mse;
  real    vec_max_err;

  // Per-element tracking
  real    abs_err;
  real    hw_val;
  real    exp_val;

  //--------------------------------------------------------------------------
  // Main test process
  //--------------------------------------------------------------------------
  integer i, v;
  integer cycle_cnt;
  integer input_offset;

  initial begin
    $display("============================================================");
    $display("  BERT Softmax Integration Testbench");
    $display("  VEC_LEN=%0d, NUM_VECTORS=%0d", VEC_LEN, NUM_VECTORS);
    $display("============================================================");
    $display("");

    // Initialize
    rst_n       = 0;
    start       = 0;
    in_valid    = 0;
    in_data     = 0;
    vec_len_cfg = VEC_LEN;
    out_idx     = 0;
    pass_cnt    = 0;
    fail_cnt    = 0;
    max_abs_error = 0.0;
    sum_sq_error  = 0.0;
    total_elements = 0.0;

    // Reset
    repeat(10) @(posedge clk);
    rst_n = 1;
    repeat(5) @(posedge clk);

    // Process each vector
    for (v = 0; v < NUM_VECTORS; v = v + 1) begin
      input_offset = v * VEC_LEN;
      vec_mse     = 0.0;
      vec_max_err = 0.0;
      vec_out_cnt = 0;

      $display("--- Vector %0d ---", v);

      // Assert start
      @(posedge clk);
      start <= 1;
      @(posedge clk);
      start <= 0;

      // Wait for in_ready and stream input data
      repeat(2) @(posedge clk); // Allow FSM to transition

      for (i = 0; i < VEC_LEN; i = i + 1) begin
        @(posedge clk);
        // Wait for in_ready if needed
        while (!in_ready) @(posedge clk);
        in_valid <= 1;
        in_data  <= all_inputs[input_offset + i];
      end
      @(posedge clk);
      in_valid <= 0;

      // Wait for all outputs or done signal
      cycle_cnt = 0;
      while (!done && cycle_cnt < TIMEOUT) begin
        @(posedge clk);
        cycle_cnt = cycle_cnt + 1;

        // Capture output
        if (out_valid && out_idx < TOTAL_ELEMS) begin
          hw_val  = real'(out_data) / real'(1 << FRAC_NORM);
          exp_val = real'(all_expected[out_idx]) / real'(1 << FRAC_NORM);
          abs_err = (hw_val > exp_val) ? (hw_val - exp_val) : (exp_val - hw_val);

          sum_sq_error = sum_sq_error + abs_err * abs_err;
          total_elements = total_elements + 1.0;

          if (abs_err > max_abs_error)
            max_abs_error = abs_err;

          vec_mse = vec_mse + abs_err * abs_err;
          if (abs_err > vec_max_err)
            vec_max_err = abs_err;

          vec_out_cnt = vec_out_cnt + 1;
          out_idx     = out_idx + 1;
        end
      end

      if (cycle_cnt >= TIMEOUT) begin
        $display("  TIMEOUT! Vector %0d did not complete.", v);
        fail_cnt = fail_cnt + 1;
      end else begin
        vec_mse = vec_mse / VEC_LEN;
        $display("  Outputs: %0d, MSE: %e, MaxErr: %.4f, Cycles: %0d",
                 vec_out_cnt, vec_mse, vec_max_err, cycle_cnt);

        if (vec_max_err < 0.01) begin
          $display("  >>> PASS <<<");
          pass_cnt = pass_cnt + 1;
        end else begin
          $display("  >>> FAIL (MaxErr > 0.01) <<<");
          fail_cnt = fail_cnt + 1;
        end
      end

      // Wait before next vector
      repeat(10) @(posedge clk);
    end

    // ======================================================================
    // Final Summary
    // ======================================================================
    $display("");
    $display("============================================================");
    $display("  Integration Test Summary");
    $display("============================================================");
    $display("  Vectors tested:      %0d", NUM_VECTORS);
    $display("  Vectors passed:      %0d", pass_cnt);
    $display("  Vectors failed:      %0d", fail_cnt);
    $display("  Total elements:      %0d", $rtoi(total_elements));
    if (total_elements > 0) begin
      $display("  Overall MSE:         %e", sum_sq_error / total_elements);
      $display("  Overall Max Error:   %.6f", max_abs_error);
    end
    $display("============================================================");

    if (fail_cnt == 0)
      $display("  >>> ALL VECTORS PASSED <<<");
    else
      $display("  >>> SOME VECTORS FAILED <<<");

    $finish;
  end

  //--------------------------------------------------------------------------
  // Watchdog timer
  //--------------------------------------------------------------------------
  initial begin
    #(TIMEOUT * CLK_PERIOD);
    $display("GLOBAL TIMEOUT REACHED — aborting simulation!");
    $finish;
  end

endmodule
