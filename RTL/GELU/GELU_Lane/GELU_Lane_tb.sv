`timescale 1ns/1ps

module GELU_Lane_tb;

  parameter int Q = 26;
  parameter int W = 32;
  parameter int INT_WIDTH = 5;
  parameter int CLK_PERIOD = 10;
  parameter int NUM_SEGMENTS = 8;

  // DUT signals
  logic                 clk;
  logic                 rst_n;
  logic                 valid_in;
  logic signed [W-1:0]  xi;
  logic [2:0]           segment_index_1;
  logic signed [W-1:0]  k_coeff_1;
  logic signed [W-1:0]  b_intercept_1;
  logic [2:0]           segment_index_2;
  logic signed [W-1:0]  k_coeff_2;
  logic signed [W-1:0]  b_intercept_2;
  logic                 valid_out;
  logic signed [W-1:0]  gelu_result;
  logic                 div_by_zero;      // NEW SIGNAL for zero-denominator flag

  // Test statistics
  int total_tests = 0;
  int passed_tests = 0;
  int failed_tests = 0;
  real max_error = 0.0;
  real max_error_input = 0.0;

  // ============================================================================
  // Instantiate ExpLUT for both EU1 and EU2
  // ============================================================================
  logic [2:0] segment_indices [1:0];
  logic signed [W-1:0] k_coeffs [1:0];
  logic signed [W-1:0] b_intercepts [1:0];

  assign segment_indices[0] = segment_index_1;
  assign segment_indices[1] = segment_index_2;
  assign k_coeff_1 = k_coeffs[0];
  assign b_intercept_1 = b_intercepts[0];
  assign k_coeff_2 = k_coeffs[1];
  assign b_intercept_2 = b_intercepts[1];

  ExpLUT #(
    .Q(Q),
    .W(W),
    .NUM_SEGMENTS(NUM_SEGMENTS),
    .NUM_PORTS(2)
  ) lut_inst (
    .segment_index(segment_indices),
    .k_coeff(k_coeffs),
    .b_intercept(b_intercepts)
  );

  // ============================================================================
  // Instantiate GELU_Lane DUT (updated port list)
  // ============================================================================
  GELU_Lane #(
    .Q(Q),
    .W(W),
    .INT_WIDTH(INT_WIDTH)
  ) dut (
    .clk(clk),
    .rst_n(rst_n),
    .valid_in(valid_in),
    .xi(xi),
    .segment_index_1(segment_index_1),
    .k_coeff_1(k_coeff_1),
    .b_intercept_1(b_intercept_1),
    .segment_index_2(segment_index_2),
    .k_coeff_2(k_coeff_2),
    .b_intercept_2(b_intercept_2),
    .valid_out(valid_out),
    .gelu_result(gelu_result),
    .div_by_zero(div_by_zero)  // NEW OUTPUT CONNECTION
  );

  // ============================================================================
  // Helper Functions
  // ============================================================================
  function real q5_26_to_real(logic signed [W-1:0] val);
    q5_26_to_real = real'(val) / (2.0 ** Q);
  endfunction

  function logic signed [W-1:0] real_to_q5_26(real val);
    real_to_q5_26 = $rtoi(val * (2.0 ** Q));
  endfunction

  // Golden GELU model using $pow
  function real gelu_golden(real x);
    real s_x, exp_s_x, result;
    const real K1 = -2.30220819814;
    const real K2 = 0.044715;
    s_x = K1 * (x + K2 * x * x * x);
    exp_s_x = $pow(2.0, s_x);
    result = x / (1.0 + exp_s_x);
    return result;
  endfunction

  // ============================================================================
  // Clock Generation
  // ============================================================================
  initial begin
    clk = 0;
    forever #(CLK_PERIOD/2) clk = ~clk;
  end

  // ============================================================================
  // Test Stimulus - Challenging Test Cases
  // ============================================================================
  initial begin
    rst_n = 0;
    valid_in = 0;
    xi = 0;

    repeat(3) @(posedge clk);
    rst_n = 1;
    repeat(2) @(posedge clk);

    $display("\n╔═══════════════════════════════════════════════════════════════════════╗");
    $display("║              GELU Lane Challenging Test Suite                         ║");
    $display("╚═══════════════════════════════════════════════════════════════════════╝\n");

    // === CRITICAL EDGE CASES ===
    run_test(0.0);
    run_test(0.000001);
    run_test(-0.000001);

    // === SMALL VALUES (-0.2 to 0.2) ===
    run_test(-0.2); run_test(-0.15); run_test(-0.1); run_test(-0.05);
    run_test(0.05); run_test(0.1); run_test(0.15); run_test(0.2);

    // === MEDIUM VALUES ===
    run_test(-1.0); run_test(-0.9); run_test(-0.8); run_test(-0.7);
    run_test(-0.6); run_test(0.6);  run_test(0.7);  run_test(0.8);
    run_test(0.9);  run_test(1.0);

    // === LARGE NEGATIVE VALUES ===
    run_test(-2.5); run_test(-2.25); run_test(-2.0);
    run_test(-1.75); run_test(-1.5); run_test(-1.25);

    // === LARGE POSITIVE VALUES ===
    run_test(1.25); run_test(1.5); run_test(1.75);
    run_test(2.0); run_test(2.25); run_test(2.5);
    run_test(2.75); run_test(3.0);

    // === EXTREME VALUES ===
    run_test(-4.0); run_test(-3.5); run_test(-3.0);
    run_test(3.5); run_test(4.0); run_test(5.0);

    // === FRACTIONAL TESTS ===
    run_test(0.333333); run_test(0.666666);
    run_test(1.414213); run_test(1.732050);
    run_test(2.718281); run_test(3.141592);
    run_test(-1.414213); run_test(-2.718281);

    // === Summary ===
    $display("\n╔═══════════════════════════════════════════════════════════════════════╗");
    $display("║                         TEST SUMMARY                                 ║");
    $display("╚═══════════════════════════════════════════════════════════════════════╝");
    $display("  Total Tests:     %0d", total_tests);
    $display("  Passed:          %0d", passed_tests);
    $display("  Failed:          %0d", failed_tests);
    $display("  Pass Rate:       %.2f%%", (real'(passed_tests) / real'(total_tests)) * 100.0);
    $display("  Max Error:       %.6f (at x = %.6f)\n", max_error, max_error_input);

    repeat(10) @(posedge clk);
    $finish;
  end

  // ============================================================================
  // Task: Run a single test with diagnostics
  // ============================================================================
  task run_test(real x_val);
    real golden_result;
    real actual_result;
    real error_abs;
    real error_pct;
    int cycles_count;
    string status_str;

    begin
      total_tests++;
      xi = real_to_q5_26(x_val);
      golden_result = gelu_golden(x_val);

      @(posedge clk);
      valid_in = 1;
      @(posedge clk);
      valid_in = 0;

      cycles_count = 0;
      while (!valid_out && cycles_count < 60) begin
        @(posedge clk);
        cycles_count++;
      end

      if (valid_out) begin
        actual_result = q5_26_to_real(gelu_result);
        error_abs = actual_result - golden_result;
        if (golden_result != 0.0)
          error_pct = (error_abs / golden_result) * 100.0;
        else
          error_pct = 0.0;

        // Track max error
        if ((error_abs > max_error) || (error_abs < -max_error)) begin
          max_error = (error_abs > 0.0) ? error_abs : -error_abs;
          max_error_input = x_val;
        end

        // Pass/fail
        if (div_by_zero) begin
          status_str = "DIV_BY_ZERO";
          failed_tests++;
        end else if (golden_result != 0.0) begin
          if ((error_pct < 10.0) && (error_pct > -10.0)) begin
            status_str = "PASS"; passed_tests++;
          end else begin
            status_str = "FAIL"; failed_tests++;
          end
        end else begin
          if ((error_abs < 0.01) && (error_abs > -0.01)) begin
            status_str = "PASS"; passed_tests++;
          end else begin
            status_str = "FAIL"; failed_tests++;
          end
        end

        $display("x=%9.6f | Actual=%9.6f | Expected=%9.6f | Err=%9.6f (%6.2f%%) | div_by_zero=%b | %s", 
                x_val, actual_result, golden_result, error_abs, error_pct, div_by_zero, status_str);
      end else begin
        $display("x=%9.6f | TIMEOUT after 60 cycles | FAIL", x_val);
        failed_tests++;
      end

      repeat(5) @(posedge clk);
    end
  endtask

  // ============================================================================
  // Waveform dumping
  // ============================================================================
  initial begin
    $dumpfile("gelu_lane_with_zero_flags.vcd");
    $dumpvars(0, GELU_Lane_tb);
  end

endmodule
