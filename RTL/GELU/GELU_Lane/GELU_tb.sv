`timescale 1ns/1ps

//=============================================================================
// GELU Testbench - Mixed Format Version
// Input: 32-bit Q10.22, Output: 64-bit Q48.16
// Tests complete GELU pipeline: Poly → EU1 → DU → EU2
// Input range: [-6.4586, +5.6423]
//=============================================================================
module GELU_tb;

    // Parameters
    localparam int Q_IN  = 22;     // Input: Q10.22
    localparam int Q_OUT = 16;     // Output: Q48.16
    localparam int W     = 32;     // 32-bit I/O width

    // Fixed-point conversion constants
    localparam real Q_SCALE_IN  = 2.0 ** Q_IN;   // 2^22 for Q10.22 input
    localparam real Q_SCALE_OUT = 2.0 ** Q_OUT;  // 2^16 for Q48.16 output

    // DUT signals
    logic signed [W-1:0]   x;
    wire  signed [2*W-1:0] y;

    // LUT interface signals
    wire [2:0]           segment_index_0;
    wire [2:0]           segment_index_1;
    wire signed [W-1:0]  k_coeff_0;
    wire signed [W-1:0]  b_intercept_0;
    wire signed [W-1:0]  k_coeff_1;
    wire signed [W-1:0]  b_intercept_1;

    // SharedLUT wiring
    wire [2:0]          segment_indices [1:0];
    wire signed [W-1:0] k_coeffs        [1:0];
    wire signed [W-1:0] b_intercepts    [1:0];

    assign segment_indices[0] = segment_index_0;
    assign segment_indices[1] = segment_index_1;
    assign k_coeff_0          = k_coeffs[0];
    assign b_intercept_0      = b_intercepts[0];
    assign k_coeff_1          = k_coeffs[1];
    assign b_intercept_1      = b_intercepts[1];

    SharedLUT #(
        .Q(22), .W(32), .NUM_SEGMENTS(8), .NUM_PORTS(2)
    ) lut_inst (
        .segment_index(segment_indices),
        .k_coeff      (k_coeffs),
        .b_intercept  (b_intercepts)
    );

    // Instantiate DUT
    GELU #(
        .Q(Q_IN), .W(W), .LUT_PORT_BASE(0)
    ) dut (
        .x             (x),
        .y             (y),
        .segment_index_0(segment_index_0),
        .segment_index_1(segment_index_1),
        .k_coeff_0     (k_coeff_0),
        .b_intercept_0 (b_intercept_0),
        .k_coeff_1     (k_coeff_1),
        .b_intercept_1 (b_intercept_1)
    );

    // =========================================================================
    // Helper Functions
    // =========================================================================

    // Convert real → Q10.22
    function automatic logic signed [W-1:0] real_to_fixed(real value);
        real scaled;
        scaled = value * Q_SCALE_IN;
        if      (scaled >  2147483647.0) return 32'sh7FFFFFFF;
        else if (scaled < -2147483648.0) return 32'sh80000000;
        else                             return $rtoi(scaled);
    endfunction

    // Convert Q48.16 → real
    function automatic real fixed_to_real(logic signed [2*W-1:0] value);
        return $itor(value) / Q_SCALE_OUT;
    endfunction

    // Absolute value
    function automatic real abs_real(real value);
        return (value < 0.0) ? -value : value;
    endfunction

    // =========================================================================
    // Golden Model 1: x / (1 + exp(-2*1.702*x))
    // Based on 1.702x sigmoid approximation
    // =========================================================================
    function automatic real gelu_golden_1(real x_val);
        real h_x, exp_term;
        h_x      = 1.702 * x_val;
        exp_term = 2.71828182845904523536 ** (-2.0 * h_x);
        return x_val / (1.0 + exp_term);
    endfunction

    // =========================================================================
    // Golden Model 2: Standard GELU (PyTorch/TensorFlow reference)
    // g(x) = 0.5*x*(1 + tanh(sqrt(2/π)*(x + 0.044715*x³)))
    // =========================================================================
    function automatic real gelu_golden_2(real x_val);
        real sqrt_2_over_pi, x_cubed, inner;
        real tanh_val, exp_pos, exp_neg;
        sqrt_2_over_pi = 0.7978845608028654;
        x_cubed        = x_val * x_val * x_val;
        inner          = sqrt_2_over_pi * (x_val + 0.044715 * x_cubed);
        exp_pos        = 2.71828182845904523536 **  inner;
        exp_neg        = 2.71828182845904523536 ** -inner;
        tanh_val       = (exp_pos - exp_neg) / (exp_pos + exp_neg);
        return 0.5 * x_val * (1.0 + tanh_val);
    endfunction

    // =========================================================================
    // Golden Model 3: Hardware-Equivalent (mirrors RTL pipeline exactly)
    //   Step 1 - PolynomialUnit : s(x) = -2.3125*(x + 0.046875*x³)
    //   Step 2 - EU1            : e    = 2^s(x)
    //   Step 3 - DU+EU2 ideal  : y    = x / (1 + e)
    // This is what a perfect (zero quantisation error) implementation
    // of the chosen architecture should output.
    // =========================================================================
    function automatic real gelu_golden_hw(real x_val);
        real s_x, exp_term, denom;
        s_x      = -2.3125 * (x_val + 0.046875 * x_val * x_val * x_val);
        exp_term =  2.0 ** s_x;            // base-2 exp, matching EU design
        denom    =  1.0 + exp_term;
        if (denom == 0.0)
            return 0.0;
        else
            return x_val / denom;
    endfunction

    // =========================================================================
    // Test statistics
    // =========================================================================
    int  pass_count_g1  = 0, pass_count_g2  = 0, pass_count_hw  = 0;
    int  fail_count_g1  = 0, fail_count_g2  = 0, fail_count_hw  = 0;
    int  total_count    = 0;
    real max_error_g1   = 0.0, max_error_g2  = 0.0, max_error_hw  = 0.0;
    real sum_error_g1   = 0.0, sum_error_g2  = 0.0, sum_error_hw  = 0.0;
    string max_error_test_g1 = "", max_error_test_g2 = "", max_error_test_hw = "";

    // =========================================================================
    // Test execution task
    // =========================================================================
    task automatic run_test(string test_name, real x_val);
        real y_golden_1, y_golden_2, y_golden_hw;
        real y_actual;
        real error_abs_g1, error_abs_g2, error_abs_hw;
        real error_pct_g1, error_pct_g2, error_pct_hw;
        string status_g1, status_g2, status_hw;

        // Drive input
        x = real_to_fixed(x_val);
        #50;

        // Compute all golden models
        y_golden_1  = gelu_golden_1 (x_val);
        y_golden_2  = gelu_golden_2 (x_val);
        y_golden_hw = gelu_golden_hw(x_val);

        // Read DUT output
        y_actual = fixed_to_real(y);

        // ------- Golden 1 errors -------
        error_abs_g1 = y_actual - y_golden_1;
        if      (y_golden_1 != 0.0) error_pct_g1 = (error_abs_g1 / y_golden_1) * 100.0;
        else if (y_actual   != 0.0) error_pct_g1 = 100.0;
        else                        error_pct_g1 = 0.0;

        // ------- Golden 2 errors -------
        error_abs_g2 = y_actual - y_golden_2;
        if      (y_golden_2 != 0.0) error_pct_g2 = (error_abs_g2 / y_golden_2) * 100.0;
        else if (y_actual   != 0.0) error_pct_g2 = 100.0;
        else                        error_pct_g2 = 0.0;

        // ------- Golden HW errors -------
        error_abs_hw = y_actual - y_golden_hw;
        if      (y_golden_hw != 0.0) error_pct_hw = (error_abs_hw / y_golden_hw) * 100.0;
        else if (y_actual    != 0.0) error_pct_hw = 100.0;
        else                         error_pct_hw = 0.0;

        // ------- Statistics -------
        total_count++;
        sum_error_g1 += abs_real(error_pct_g1);
        sum_error_g2 += abs_real(error_pct_g2);
        sum_error_hw += abs_real(error_pct_hw);

        if (abs_real(error_pct_g1) > max_error_g1) begin
            max_error_g1      = abs_real(error_pct_g1);
            max_error_test_g1 = test_name;
        end
        if (abs_real(error_pct_g2) > max_error_g2) begin
            max_error_g2      = abs_real(error_pct_g2);
            max_error_test_g2 = test_name;
        end
        if (abs_real(error_pct_hw) > max_error_hw) begin
            max_error_hw      = abs_real(error_pct_hw);
            max_error_test_hw = test_name;
        end

        // ------- Pass/fail decisions -------
        // G1/G2 : 30% relative OR |abs| < 0.05  (Mitchell approximation budget)
        if (abs_real(error_pct_g1) < 30.0 || abs_real(error_abs_g1) < 0.05) begin
            status_g1 = "✓ PASS"; pass_count_g1++;
        end else begin
            status_g1 = "✗ FAIL"; fail_count_g1++;
        end

        if (abs_real(error_pct_g2) < 30.0 || abs_real(error_abs_g2) < 0.05) begin
            status_g2 = "✓ PASS"; pass_count_g2++;
        end else begin
            status_g2 = "✗ FAIL"; fail_count_g2++;
        end

        // HW model: 10% relative OR |abs| < 0.02  (tight: same architecture)
        if (abs_real(error_pct_hw) < 10.0 || abs_real(error_abs_hw) < 0.02) begin
            status_hw = "✓ PASS"; pass_count_hw++;
        end else begin
            status_hw = "✗ FAIL"; fail_count_hw++;
        end

        // ------- Display -------
        $display("╔════════════════════════════════════════════════════════════════════════════╗");
        $display("║ TEST #%-2d: %-64s ║", total_count, test_name);
        $display("╠════════════════════════════════════════════════════════════════════════════╣");
        $display("║ Input (x):           %12.6f  (0x%08h Q10.22)                  ║", x_val, x);
        $display("║                                                                            ║");
        $display("║ Golden Model 1 [x/(1+exp(-2*1.702x))]:                                    ║");
        $display("║   Value:             %12.6f                                      ║", y_golden_1);
        $display("║   Error (abs):       %12.6f                                      ║", error_abs_g1);
        $display("║   Error (%%):         %12.6f%%                                     ║", error_pct_g1);
        $display("║   Status:            %-52s ║", status_g1);
        $display("║                                                                            ║");
        $display("║ Golden Model 2 [0.5x(1+tanh(...))]:                                       ║");
        $display("║   Value:             %12.6f                                      ║", y_golden_2);
        $display("║   Error (abs):       %12.6f                                      ║", error_abs_g2);
        $display("║   Error (%%):         %12.6f%%                                     ║", error_pct_g2);
        $display("║   Status:            %-52s ║", status_g2);
        $display("║                                                                            ║");
        $display("║ Golden Model HW [x/(1+2^s(x)), s=-2.3125(x+0.047x³)]:                    ║");
        $display("║   Value:             %12.6f                                      ║", y_golden_hw);
        $display("║   Error (abs):       %12.6f                                      ║", error_abs_hw);
        $display("║   Error (%%):         %12.6f%%                                     ║", error_pct_hw);
        $display("║   Status:            %-52s ║", status_hw);
        $display("║                                                                            ║");
        $display("║ Actual Output (y):   %12.6f  (0x%016h Q48.16)          ║", y_actual, y);
        $display("║ LUT indices:         EU1=%0d, EU2=%0d                                      ║", segment_index_0, segment_index_1);
        $display("╚════════════════════════════════════════════════════════════════════════════╝\n");

        #10;
    endtask

    // =========================================================================
    // Summary task
    // =========================================================================
    task display_summary();
        real pass_rate_g1, pass_rate_g2, pass_rate_hw;
        real avg_error_g1, avg_error_g2, avg_error_hw;

        pass_rate_g1 = (real'(pass_count_g1) / real'(total_count)) * 100.0;
        pass_rate_g2 = (real'(pass_count_g2) / real'(total_count)) * 100.0;
        pass_rate_hw = (real'(pass_count_hw) / real'(total_count)) * 100.0;
        avg_error_g1 = sum_error_g1 / real'(total_count);
        avg_error_g2 = sum_error_g2 / real'(total_count);
        avg_error_hw = sum_error_hw / real'(total_count);

        $display("║                           TEST SUMMARY                                     ║");
        $display("║                                                                            ║");
        $display("║ Total Tests:     %3d                                                       ║", total_count);
        $display("║ Input Format:    32-bit Q10.22                                             ║");
        $display("║ Output Format:   64-bit Q48.16                                             ║");
        $display("║ Input Range:     [-6.4586, +5.6423]                                       ║");
        $display("║                                                                            ║");
        $display("║ ══════════════════════════════════════════════════════════════════════════ ║");
        $display("║ Golden 1 : x / (1 + exp(-2*1.702*x))    [tolerance: 30%% or |e|<0.05]     ║");
        $display("║ ══════════════════════════════════════════════════════════════════════════ ║");
        $display("║   Passed:        %3d / %3d   Pass Rate: %6.2f%%                           ║", pass_count_g1, total_count, pass_rate_g1);
        $display("║   Max Error:     %6.2f%% (test: %-30s)            ║", max_error_g1, max_error_test_g1);
        $display("║   Avg Error:     %6.2f%%                                                   ║", avg_error_g1);
        $display("║                                                                            ║");
        $display("║ ══════════════════════════════════════════════════════════════════════════ ║");
        $display("║ Golden 2 : 0.5x(1+tanh(sqrt(2/π)(x+0.044715x³)))  [tol: 30%% or |e|<0.05]║");
        $display("║ ══════════════════════════════════════════════════════════════════════════ ║");
        $display("║   Passed:        %3d / %3d   Pass Rate: %6.2f%%                           ║", pass_count_g2, total_count, pass_rate_g2);
        $display("║   Max Error:     %6.2f%% (test: %-30s)            ║", max_error_g2, max_error_test_g2);
        $display("║   Avg Error:     %6.2f%%                                                   ║", avg_error_g2);
        $display("║                                                                            ║");
        $display("║ ══════════════════════════════════════════════════════════════════════════ ║");
        $display("║ Golden HW: x/(1+2^s(x)), s=-2.3125(x+0.047x³)   [tolerance: 10%% or |e|<0.02] ║");
        $display("║ ══════════════════════════════════════════════════════════════════════════ ║");
        $display("║   Passed:        %3d / %3d   Pass Rate: %6.2f%%                           ║", pass_count_hw, total_count, pass_rate_hw);
        $display("║   Max Error:     %6.2f%% (test: %-30s)            ║", max_error_hw, max_error_test_hw);
        $display("║   Avg Error:     %6.2f%%                                                   ║", avg_error_hw);
        $display("║                                                                            ║");

        if (fail_count_hw == 0) begin
            $display("║ ✅ HW MODEL: ALL TESTS PASSED - Architecture verified!                    ║");
        end else begin
            $display("║ ⚠️  HW MODEL: %2d tests failed - RTL deviates from intended function.      ║", fail_count_hw);
        end

        if (fail_count_g1 == 0 && fail_count_g2 == 0) begin
            $display("║ ✅ GELU REF: ALL TESTS PASSED - Matches standard GELU!                   ║");
        end else begin
            $display("║ ℹ️  GELU REF: %2d/%2d G1, %2d/%2d G2 passed (approximation delta expected). ║",
                     pass_count_g1, total_count, pass_count_g2, total_count);
        end
    endtask

    // =========================================================================
// Debug probe task - add intermediate signal monitors
// =========================================================================
task automatic debug_test(string test_name, real x_val);
    // Drive input
    x = real_to_fixed(x_val);
    #50;

    $display("════════ DEBUG: %s (x = %f) ════════", test_name, x_val);
    $display("  x_q1022       = 0x%08h  = %f", x, $itor(signed'(x)) / (2.0**22));
    
    // Peek internal DUT nets (requires -access +r in QuestaSim or direct hierarchy)
    $display("  s_x_q1022     = 0x%08h  = %f", dut.s_x_q1022,
             $itor(signed'(dut.s_x_q1022)) / (2.0**22));
    
    $display("  exp_s_x[63:0] = 0x%016h  = %f", dut.exp_s_x,
             $itor(signed'(dut.exp_s_x)) / (2.0**16));
    
    $display("  x_q4816       = 0x%016h  = %f", dut.x_q4816,
             $itor(signed'(dut.x_q4816)) / (2.0**16));
    
    $display("  du_exponent   = 0x%016h  = %f", dut.du_exponent,
             $itor(signed'(dut.du_exponent)) / (2.0**16));
    
    $display("  du_sign       = %b", dut.du_sign);
    
    $display("  du_exp_q1022  = 0x%08h  = %f", dut.du_exp_q1022,
             $itor(signed'(dut.du_exp_q1022)) / (2.0**22));
    
    $display("  eu2_result    = 0x%016h  = %f", dut.eu2_result_64,
             $itor(signed'(dut.eu2_result_64)) / (2.0**16));
    
    $display("  y (output)    = 0x%016h  = %f", y,
             $itor(signed'(y)) / (2.0**16));
    $display("");
    
    #10;
endtask

    // =========================================================================
    // Test Procedure
    // =========================================================================
    initial begin
        $display("\n╔════════════════════════════════════════════════════════════════════════════╗");
        $display("║                   GELU MODULE TESTBENCH - MIXED FORMAT                     ║");
        $display("║               Flow: x(Q10.22) → Poly → EU1 → DU → EU2 → y(Q48.16)        ║");
        $display("║                                                                            ║");
        $display("║  Golden 1: x / (1 + exp(-2*1.702*x))                                      ║");
        $display("║  Golden 2: 0.5*x*(1 + tanh(sqrt(2/π)*(x + 0.044715*x³)))                  ║");
        $display("║  Golden HW: x / (1 + 2^(-2.3125*(x+0.046875*x³)))   ← RTL match          ║");
        $display("║                                                                            ║");
        $display("║  Tolerance G1/G2: 30%% or |abs|<0.05                                       ║");
        $display("║  Tolerance HW:    10%% or |abs|<0.02                                       ║");
        $display("╚════════════════════════════════════════════════════════════════════════════╝\n");

        #10;

        // ── Boundary ──────────────────────────────────────────────────────────
        $display("════════ BOUNDARY VALUES ════════\n");
        run_test("Min boundary",  -6.4586);
        run_test("Max boundary",   5.6423);
        run_test("Zero",           0.0);

        // ── Positive ──────────────────────────────────────────────────────────
        $display("\n════════ POSITIVE VALUES [0, 5.6423] ════════\n");
        run_test("Very small pos",  0.01);
        run_test("Tiny pos",        0.1);
        run_test("Quarter",         0.25);
        run_test("Half",            0.5);
        run_test("Three quarters",  0.75);
        run_test("Unity",           1.0);
        run_test("1.25",            1.25);
        run_test("1.5",             1.5);
        run_test("1.75",            1.75);
        run_test("Two",             2.0);
        run_test("2.5",             2.5);
        run_test("Three",           3.0);
        run_test("3.5",             3.5);
        run_test("Four",            4.0);
        run_test("4.5",             4.5);
        run_test("Five",            5.0);
        run_test("5.5",             5.5);
        run_test("Near max",        5.6);

        // ── Negative ──────────────────────────────────────────────────────────
        $display("\n════════ NEGATIVE VALUES [-6.4586, 0] ════════\n");
        run_test("Very small neg",       -0.01);
        run_test("Tiny neg",             -0.1);
        run_test("Minus quarter",        -0.25);
        run_test("Minus half",           -0.5);
        run_test("Minus three quarters", -0.75);
        run_test("Minus one",            -1.0);
        run_test("-1.25",                -1.25);
        run_test("-1.5",                 -1.5);
        run_test("-1.75",                -1.75);
        run_test("Minus two",            -2.0);
        run_test("-2.5",                 -2.5);
        run_test("Minus three",          -3.0);
        run_test("-3.5",                 -3.5);
        run_test("Minus four",           -4.0);
        run_test("-4.5",                 -4.5);
        run_test("Minus five",           -5.0);
        run_test("-5.5",                 -5.5);
        run_test("Minus six",            -6.0);
        run_test("Near min",             -6.4);

        // ── Critical transitions ───────────────────────────────────────────────
        $display("\n════════ CRITICAL TRANSITION POINTS ════════\n");
        run_test("Transition +0.84",  0.8414);
        run_test("Transition -0.84", -0.8414);
        run_test("99% sat +",         2.5);
        run_test("99% sat -",        -2.5);
        run_test("Linear region +",   0.3);
        run_test("Linear region -",  -0.3);

        // ── Dense sweep ───────────────────────────────────────────────────────
        $display("\n════════ DENSE SWEEP [-6 to +6, step 0.5] ════════\n");
        for (real val = -6.0; val <= 6.0; val += 0.5) begin
            if (val >= -6.4586 && val <= 5.6423)
                run_test($sformatf("Sweep %.1f", val), val);
        end

        // ── Out-of-range ──────────────────────────────────────────────────────
        $display("\n════════ OUT-OF-RANGE (Saturation) ════════\n");
        run_test("Beyond max +7",  7.0);
        run_test("Beyond max +10", 10.0);
        run_test("Beyond min -7",  -7.0);
        run_test("Beyond min -10", -10.0);

        $display("\n╔════════════════════════════════════════════════════════════════════════════╗");
        $display("║                         ALL TESTS COMPLETED                                ║");
        $display("║                                                                            ║");
        display_summary();
        $display("╚════════════════════════════════════════════════════════════════════════════╝\n");


        #10;
        // ── Targeted debug ────────────────────────────────────────────────────
        $display("\n════════ STAGE-BY-STAGE DEBUG ════════\n");
        debug_test("DEBUG pos +1.0",  1.0);   // known passing case
        debug_test("DEBUG neg -1.0", -1.0);   // known failing case
        debug_test("DEBUG neg -0.5", -0.5);   // failing

        $finish;
    end

endmodule
