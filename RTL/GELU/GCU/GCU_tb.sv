`timescale 1ns/1ps

//=============================================================================
// GCU Testbench - 32 Parallel GELU Lanes
// Input:  32 × 32-bit Q10.22
// Output: 32 × 64-bit Q48.16
//=============================================================================
module GCU_tb;

    localparam int Q_IN      = 22;
    localparam int Q_OUT     = 16;
    localparam int W         = 32;
    localparam int NUM_GELU  = 32;

    localparam real Q_SCALE_IN  = 2.0 ** Q_IN;
    localparam real Q_SCALE_OUT = 2.0 ** Q_OUT;

    logic signed [W-1:0]   x [NUM_GELU-1:0];
    wire  signed [2*W-1:0] y [NUM_GELU-1:0];

    GCU #(
        .Q(Q_IN),
        .W(W),
        .NUM_GELU(NUM_GELU),
        .NUM_LUT_PORTS(64)
    ) dut (.x(x), .y(y));

    // =========================================================================
    // Helper Functions
    // =========================================================================

    function automatic logic signed [W-1:0] real_to_fixed(real value);
        real scaled;
        scaled = value * Q_SCALE_IN;
        if      (scaled >  2147483647.0) return 32'sh7FFFFFFF;
        else if (scaled < -2147483648.0) return 32'sh80000000;
        else                             return $rtoi(scaled);
    endfunction

    function automatic real fixed_to_real(logic signed [2*W-1:0] value);
        return $itor(value) / Q_SCALE_OUT;
    endfunction

    function automatic real abs_real(real value);
        return (value < 0.0) ? -value : value;
    endfunction

    // ✅ Flush: kills -0.0 and any sub-LSB noise (Q48.16 LSB ≈ 1.5e-5)
    function automatic real flush(real value);
        return (abs_real(value) < 1e-6) ? 0.0 : value;
    endfunction

    // ✅ Safe error %: flushes both operands before dividing
    function automatic real calc_pct(real actual, real golden);
        real a, g, err;
        a = flush(actual);
        g = flush(golden);
        err = a - g;
        if      (abs_real(g) > 1e-6) return (err / g) * 100.0;
        else if (abs_real(a) > 1e-6) return 100.0;
        else                         return 0.0;  // both ~0 → 0% error
    endfunction

    function automatic real gelu_golden_hw(real x_val);
        real s_x, exp_term, denom;
        if (x_val == 0.0) return 0.0;
        s_x      = -2.3125 * (x_val + 0.046875 * x_val * x_val * x_val);
        exp_term =  2.0 ** s_x;
        denom    =  1.0 + exp_term;
        if (denom == 0.0) return 0.0;
        return flush(x_val / denom);
    endfunction

    function automatic real gelu_golden_2(real x_val);
        real sqrt_2_over_pi, x_cubed, inner, tanh_val, exp_pos, exp_neg;
        if (x_val == 0.0) return 0.0;
        sqrt_2_over_pi = 0.7978845608028654;
        x_cubed        = x_val * x_val * x_val;
        inner          = sqrt_2_over_pi * (x_val + 0.044715 * x_cubed);
        exp_pos        = 2.71828182845904523536 **  inner;
        exp_neg        = 2.71828182845904523536 ** -inner;
        tanh_val       = (exp_pos - exp_neg) / (exp_pos + exp_neg);
        return flush(0.5 * x_val * (1.0 + tanh_val));
    endfunction

    // =========================================================================
    // Statistics
    // =========================================================================
    int  total_lane_tests = 0;
    int  pass_hw = 0, fail_hw = 0;
    int  pass_g2 = 0, fail_g2 = 0;
    real max_error_hw = 0.0, max_error_g2 = 0.0;
    real sum_error_hw = 0.0, sum_error_g2 = 0.0;
    string max_err_test_hw = "", max_err_test_g2 = "";

    // =========================================================================
    // Task: All 32 lanes same value
    // =========================================================================
    task automatic run_uniform_test(string test_name, real x_val);
        real y_hw, y_g2, y_actual;
        real err_hw, err_g2, pct_hw, pct_g2;
        int  lane_fail_hw, lane_fail_g2;
        lane_fail_hw = 0;
        lane_fail_g2 = 0;

        for (int i = 0; i < NUM_GELU; i++)
            x[i] = real_to_fixed(x_val);
        #50;

        y_hw = gelu_golden_hw(x_val);
        y_g2 = gelu_golden_2(x_val);

        for (int lane = 0; lane < NUM_GELU; lane++) begin
            y_actual = fixed_to_real(y[lane]);

            pct_hw = calc_pct(y_actual, y_hw);
            pct_g2 = calc_pct(y_actual, y_g2);
            err_hw = flush(y_actual) - y_hw;
            err_g2 = flush(y_actual) - y_g2;

            total_lane_tests++;

            // ✅ Only accumulate stats when at least one side is non-trivial
            if (abs_real(flush(y_actual)) > 1e-6 || abs_real(y_hw) > 1e-6) begin
                sum_error_hw += abs_real(pct_hw);
                if (abs_real(pct_hw) > max_error_hw) begin
                    max_error_hw    = abs_real(pct_hw);
                    max_err_test_hw = test_name;
                end
            end

            if (abs_real(flush(y_actual)) > 1e-6 || abs_real(y_g2) > 1e-6) begin
                sum_error_g2 += abs_real(pct_g2);
                if (abs_real(pct_g2) > max_error_g2) begin
                    max_error_g2    = abs_real(pct_g2);
                    max_err_test_g2 = test_name;
                end
            end

            // HW: 10% or |abs| < 0.02
            if (abs_real(pct_hw) < 10.0 || abs_real(err_hw) < 0.02)
                pass_hw++;
            else begin
                fail_hw++;
                lane_fail_hw++;
                $display("  ⚠️  LANE %2d HW FAIL: x=%7.4f  y=%10.6f  exp=%10.6f  err=%.2f%%",
                         lane, x_val, y_actual, y_hw, pct_hw);
            end

            // G2: 30% or |abs| < 0.05
            if (abs_real(pct_g2) < 30.0 || abs_real(err_g2) < 0.05)
                pass_g2++;
            else begin
                fail_g2++;
                lane_fail_g2++;
                $display("  ⚠️  LANE %2d G2 FAIL: x=%7.4f  y=%10.6f  exp=%10.6f  err=%.2f%%",
                         lane, x_val, y_actual, y_g2, pct_g2);
            end
        end

        y_actual = fixed_to_real(y[0]);
        $display("  [UNIFORM] %-22s  x=%7.4f | y[0]=%10.6f | hw_exp=%10.6f | g2_exp=%10.6f | hw_fail=%0d g2_fail=%0d",
                 test_name, x_val, y_actual, y_hw, y_g2, lane_fail_hw, lane_fail_g2);
        #10;
    endtask

    // =========================================================================
    // Task: 32 unique values per lane
    // =========================================================================
    task automatic run_vector_test(string test_name, real x_vals [NUM_GELU-1:0]);
        real y_hw, y_actual, pct_hw, err_hw;
        int  lane_fail_hw;
        lane_fail_hw = 0;

        for (int i = 0; i < NUM_GELU; i++)
            x[i] = real_to_fixed(x_vals[i]);
        #50;

        $display("\n  [VECTOR]  %s", test_name);
        for (int lane = 0; lane < NUM_GELU; lane++) begin
            y_actual = fixed_to_real(y[lane]);
            y_hw     = gelu_golden_hw(x_vals[lane]);

            pct_hw = calc_pct(y_actual, y_hw);
            err_hw = flush(y_actual) - y_hw;

            total_lane_tests++;

            // ✅ Only accumulate stats when at least one side is non-trivial
            if (abs_real(flush(y_actual)) > 1e-6 || abs_real(y_hw) > 1e-6) begin
                sum_error_hw += abs_real(pct_hw);
                if (abs_real(pct_hw) > max_error_hw) begin
                    max_error_hw    = abs_real(pct_hw);
                    max_err_test_hw = test_name;
                end
            end

            if (abs_real(pct_hw) < 10.0 || abs_real(err_hw) < 0.02)
                pass_hw++;
            else begin
                fail_hw++;
                lane_fail_hw++;
                $display("    ⚠️  LANE %2d FAIL: x=%7.4f  y=%10.6f  exp=%10.6f  err=%.2f%%",
                         lane, x_vals[lane], y_actual, y_hw, pct_hw);
            end

            $display("    Lane[%2d]: x=%7.4f | y=%10.6f | hw_exp=%10.6f | err=%.3f%%",
                     lane, x_vals[lane], y_actual, y_hw, pct_hw);
        end

        if (lane_fail_hw == 0)
            $display("    ✅ All 32 lanes PASSED");
        #10;
    endtask

    // =========================================================================
    // Main Test Procedure
    // =========================================================================
    initial begin
        for (int i = 0; i < NUM_GELU; i++) x[i] = '0;
        #10;

        $display("\n╔══════════════════════════════════════════════════════════════════════╗");
        $display("║              GCU TESTBENCH - 32 Parallel GELU Lanes                ║");
        $display("║  Input:    32 × Q10.22 (32-bit)                                    ║");
        $display("║  Output:   32 × Q48.16 (64-bit)                                    ║");
        $display("║  GoldenHW: x/(1+2^s(x)), s=-2.3125*(x+0.046875*x³)                ║");
        $display("║  Golden2:  0.5x(1+tanh(sqrt(2/π)(x+0.044715x³)))                  ║");
        $display("║  Tol HW:   10%% or |abs|<0.02                                       ║");
        $display("║  Tol G2:   30%% or |abs|<0.05                                       ║");
        $display("╚══════════════════════════════════════════════════════════════════════╝\n");

        $display("════════ TEST 1: BOUNDARY VALUES ════════");
        run_uniform_test("Min boundary",  -6.4586);
        run_uniform_test("Max boundary",   5.6423);
        run_uniform_test("Zero",           0.0);

        $display("\n════════ TEST 2: POSITIVE RANGE ════════");
        run_uniform_test("0.01",   0.01);
        run_uniform_test("0.1",    0.1);
        run_uniform_test("0.25",   0.25);
        run_uniform_test("0.5",    0.5);
        run_uniform_test("0.75",   0.75);
        run_uniform_test("1.0",    1.0);
        run_uniform_test("1.5",    1.5);
        run_uniform_test("2.0",    2.0);
        run_uniform_test("2.5",    2.5);
        run_uniform_test("3.0",    3.0);
        run_uniform_test("3.5",    3.5);
        run_uniform_test("4.0",    4.0);
        run_uniform_test("4.5",    4.5);
        run_uniform_test("5.0",    5.0);
        run_uniform_test("5.5",    5.5);

        $display("\n════════ TEST 3: NEGATIVE RANGE ════════");
        run_uniform_test("-0.01",  -0.01);
        run_uniform_test("-0.1",   -0.1);
        run_uniform_test("-0.25",  -0.25);
        run_uniform_test("-0.5",   -0.5);
        run_uniform_test("-0.75",  -0.75);
        run_uniform_test("-1.0",   -1.0);
        run_uniform_test("-1.5",   -1.5);
        run_uniform_test("-2.0",   -2.0);
        run_uniform_test("-2.5",   -2.5);
        run_uniform_test("-3.0",   -3.0);
        run_uniform_test("-3.5",   -3.5);
        run_uniform_test("-4.0",   -4.0);
        run_uniform_test("-4.5",   -4.5);
        run_uniform_test("-5.0",   -5.0);
        run_uniform_test("-5.5",   -5.5);
        run_uniform_test("-6.0",   -6.0);

        $display("\n════════ TEST 4: CRITICAL TRANSITIONS ════════");
        run_uniform_test("Inflection +0.84",    0.8414);
        run_uniform_test("Inflection -0.84",   -0.8414);
        run_uniform_test("Linear region +0.3",  0.3);
        run_uniform_test("Linear region -0.3", -0.3);
        run_uniform_test("99pct sat +2.5",      2.5);
        run_uniform_test("99pct sat -2.5",     -2.5);

        $display("\n════════ TEST 5: DENSE SWEEP [-6, +5.5] step 0.5 ════════");
        for (real val = -6.0; val <= 5.5; val += 0.5)
            if (val >= -6.4586 && val <= 5.6423)
                run_uniform_test($sformatf("Sweep %.1f", val), val);

        $display("\n════════ TEST 6: FULL RANGE VECTOR (32 unique values) ════════");
        begin
            real x_vec[NUM_GELU-1:0];
            for (int i = 0; i < NUM_GELU; i++)
                x_vec[i] = -6.4586 + (12.1009 / 31.0) * i;
            run_vector_test("Full range spread [-6.46..+5.64]", x_vec);
        end

        $display("\n════════ TEST 7: ALTERNATING ±1.0 PER LANE ════════");
        begin
            real x_alt[NUM_GELU-1:0];
            for (int i = 0; i < NUM_GELU; i++)
                x_alt[i] = (i % 2 == 0) ? 1.0 : -1.0;
            run_vector_test("Alternating +1.0 / -1.0", x_alt);
        end

        $display("\n════════ TEST 8: ALTERNATING LARGE/SMALL ════════");
        begin
            real x_mix[NUM_GELU-1:0];
            for (int i = 0; i < NUM_GELU; i++)
                x_mix[i] = (i % 2 == 0) ? 5.0 : -5.0;
            run_vector_test("Alternating +5.0 / -5.0", x_mix);
        end

        $display("\n════════ TEST 9: ALL ZEROS ════════");
        run_uniform_test("All zeros", 0.0);

        $display("\n════════ TEST 10: OUT-OF-RANGE (Saturation) ════════");
        run_uniform_test("OOR +7.0",   7.0);
        run_uniform_test("OOR +10.0", 10.0);
        run_uniform_test("OOR -7.0",  -7.0);
        run_uniform_test("OOR -10.0",-10.0);

        $display("\n╔══════════════════════════════════════════════════════════════════════╗");
        $display("║                       GCU TEST SUMMARY                              ║");
        $display("╠══════════════════════════════════════════════════════════════════════╣");
        $display("║  Total Lane Tests:  %5d  (unique lane × value combinations)        ║",
                 total_lane_tests);
        $display("║                                                                      ║");
        $display("║  HW Model [tol: 10%% or |e|<0.02]:                                   ║");
        $display("║    Passed: %5d / %5d  (%6.2f%%)                                  ║",
                 pass_hw, total_lane_tests,
                 (real'(pass_hw)/real'(total_lane_tests))*100.0);
        $display("║    Failed: %5d                                                      ║", fail_hw);
        $display("║    Max Error: %7.4f%%  (at: %-28s)          ║",
                 max_error_hw, max_err_test_hw);
        $display("║    Avg Error: %7.4f%%                                               ║",
                 sum_error_hw / real'(total_lane_tests));
        $display("║                                                                      ║");
        $display("║  G2 Model [tol: 30%% or |e|<0.05]:                                   ║");
        $display("║    Passed: %5d / %5d  (%6.2f%%)                                  ║",
                 pass_g2, total_lane_tests,
                 (real'(pass_g2)/real'(total_lane_tests))*100.0);
        $display("║    Failed: %5d                                                      ║", fail_g2);
        $display("║    Max Error: %7.4f%%  (at: %-28s)          ║",
                 max_error_g2, max_err_test_g2);
        $display("║    Avg Error: %7.4f%%                                               ║",
                 sum_error_g2 / real'(total_lane_tests));
        $display("╠══════════════════════════════════════════════════════════════════════╣");

        if (fail_hw == 0)
            $display("║  ✅ ALL LANES PASSED - GCU Architecture Fully Verified!              ║");
        else
            $display("║  ⚠️  %4d lane failures - inspect per-lane debug above                ║", fail_hw);

        if (fail_g2 == 0)
            $display("║  ✅ G2 GELU REF: ALL PASSED                                          ║");
        else
            $display("║  ℹ️  G2 approx delta expected - %4d outside 30%% tol                  ║", fail_g2);

        $display("╚══════════════════════════════════════════════════════════════════════╝\n");

        $finish;
    end

endmodule
