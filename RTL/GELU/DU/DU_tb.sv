`timescale 1ns/1ps

module DU_tb;

    parameter int Q = 16;  // Q48.16 format
    parameter int W = 64;  // 64-bit width

    // DUT signals
    logic signed [W-1:0]  F;
    logic signed [W-1:0]  s_xi;
    logic signed [W-1:0]  exponent;
    logic                 result_sign;

    // Internal signals for monitoring
    logic signed [W-1:0] denominator;
    logic signed [W-1:0] m1;
    logic signed [W-1:0] m2;
    logic signed [W-1:0] s1;
    logic signed [W-1:0] s2;

    // Test tracking
    int test_count = 0;
    int pass_count = 0;
    int fail_count = 0;

    // DUT instantiation
    DU #(.Q(Q), .W(W)) dut (
        .F(F),
        .s_xi(s_xi),
        .exponent(exponent),
        .result_sign(result_sign)
    );

    // Probe internal signals
    assign denominator = dut.denominator;
    assign m1 = dut.m1;
    assign m2 = dut.m2;
    assign s1 = dut.s1;
    assign s2 = dut.s2;

    //===========================================================================
    // HELPER FUNCTIONS
    //===========================================================================

    // Convert Q48.16 to decimal
    function real q48_16_to_decimal(logic signed [W-1:0] val);
        begin
            q48_16_to_decimal = real'(val) / (real'(2) ** Q);
        end
    endfunction

    // Convert decimal to Q48.16
    function logic signed [W-1:0] decimal_to_q48_16(real val);
        real scaled;
        begin
            scaled = val * (real'(2) ** Q);
            if (scaled > 9223372036854775807.0)
                decimal_to_q48_16 = 64'sh7FFFFFFFFFFFFFFF;
            else if (scaled < -9223372036854775808.0)
                decimal_to_q48_16 = -64'sh8000000000000000;
            else
                decimal_to_q48_16 = $rtoi(scaled);
        end
    endfunction

    // 2^exponent where exponent is in Q48.16
    function real compute_2_to_exponent(logic signed [W-1:0] exp_q48_16);
        real exp_decimal;
        begin
            exp_decimal = q48_16_to_decimal(exp_q48_16);
            compute_2_to_exponent = 2.0 ** exp_decimal;
        end
    endfunction

    // Calculate error percentage
    function real calc_error_percent(real expected, real actual);
        begin
            if (expected == 0.0)
                calc_error_percent = 0.0;
            else
                calc_error_percent = ((actual - expected) / expected) * 100.0;
        end
    endfunction

    // Calculate absolute error percentage
    function real calc_abs_error_percent(real expected, real actual);
        real err;
        begin
            err = calc_error_percent(expected, actual);
            calc_abs_error_percent = (err < 0.0) ? -err : err;
        end
    endfunction

    //===========================================================================
    // TEST TASK
    //===========================================================================
    task run_test(
        input string test_name,
        input real f_val,
        input real sxi_val,
        input real tolerance_percent
    );
        real expected_result;
        real actual_magnitude;
        real error_percent;
        real abs_error;
        logic expected_sign;

        begin
            test_count++;

            // Convert to Q48.16
            F = decimal_to_q48_16(f_val);
            s_xi = decimal_to_q48_16(sxi_val);

            // Wait for combinational logic to settle
            #1;

            // Calculate expected result
            expected_result = f_val / (1.0 + sxi_val);
            expected_sign = (expected_result < 0.0);

            // Get actual result (magnitude only, sign separate)
            actual_magnitude = compute_2_to_exponent(exponent);

            // Calculate error on magnitude
            if (expected_result < 0.0)
                error_percent = calc_error_percent(-expected_result, actual_magnitude);
            else
                error_percent = calc_error_percent(expected_result, actual_magnitude);

            abs_error = (error_percent < 0.0) ? -error_percent : error_percent;

            // Display test header
            $display("\n╔════════════════════════════════════════════════════════════════════╗");
            $display("║ TEST %0d: %-60s ║", test_count, test_name);
            $display("╚════════════════════════════════════════════════════════════════════╝");

            // Display inputs
            $display("\nINPUTS:");
            $display("  F        = 0x%016h = %.10f", F, q48_16_to_decimal(F));
            $display("  s_xi     = 0x%016h = %.10f", s_xi, q48_16_to_decimal(s_xi));
            $display("  Expected denominator (1 + s_xi) = %.10f", 1.0 + q48_16_to_decimal(s_xi));
            $display("  Expected result: %.10f / %.10f = %.10f", 
                     q48_16_to_decimal(F), 1.0 + q48_16_to_decimal(s_xi), expected_result);

            // Display internal signals
            $display("\nINTERNAL SIGNALS:");
            $display("  denominator = 0x%016h = %.10f", denominator, q48_16_to_decimal(denominator));
            $display("  s1          = 0x%016h = %.10f", s1, q48_16_to_decimal(s1));
            $display("  s2          = 0x%016h = %.10f", s2, q48_16_to_decimal(s2));
            $display("  m1          = 0x%016h = %.10f", m1, q48_16_to_decimal(m1));
            $display("  m2          = 0x%016h = %.10f", m2, q48_16_to_decimal(m2));

            // Display outputs
            $display("\nFINAL OUTPUT:");
            $display("  exponent (hex)     = 0x%016h", exponent);
            $display("  exponent (decimal) = %.10f", q48_16_to_decimal(exponent));
            $display("  2^exponent         = %.10f", actual_magnitude);
            $display("  result_sign        = %b (%s)", result_sign, result_sign ? "NEGATIVE" : "POSITIVE");

            if (expected_result < 0.0) begin
                $display("  Actual result      = -%.10f", actual_magnitude);
                $display("  Expected result    = %.10f", expected_result);
            end else begin
                $display("  Actual result      = %.10f", actual_magnitude);
                $display("  Expected result    = %.10f", expected_result);
            end

            $display("  Error (magnitude)  = %.6f%%", error_percent);

            // Check sign
            if (result_sign != expected_sign) begin
                $display("  [SIGN MISMATCH] Expected sign=%b, Got sign=%b", expected_sign, result_sign);
                $display("  ❌ FAIL");
                fail_count++;
            end else if (abs_error > tolerance_percent) begin
                $display("  [ERROR TOO HIGH] Exceeds %.1f%% tolerance", tolerance_percent);
                $display("  ❌ FAIL");
                fail_count++;
            end else begin
                $display("  ✅ PASS (within %.1f%% tolerance)", tolerance_percent);
                pass_count++;
            end
        end
    endtask

    //===========================================================================
    // MAIN TEST SEQUENCE
    //===========================================================================
    initial begin
        $display("\n╔═══════════════════════════════════════════════════════════════════════╗");
        $display("║           DU Testbench - Testing F / (1 + s_xi)                      ║");
        $display("║           Format: Q48.16 (48 integer bits, 16 fractional bits)       ║");
        $display("║           Combinational Version (No Clock)                           ║");
        $display("║           Mitchell's Algorithm with ~5-20%% approximation error      ║");
        $display("╚═══════════════════════════════════════════════════════════════════════╝\n");

        // Initialize
        F = 0;
        s_xi = 0;
        #10;  // Let signals settle

        // ========== BASIC TESTS ==========
        run_test("F = 7.0, s_xi = 2.0 → 7.0/3.0 = 2.333...", 7.0, 2.0, 20.0);
        run_test("F = 11.0, s_xi = 6.0 → 11.0/7.0 = 1.571...", 11.0, 6.0, 20.0);
        run_test("F = 13.0, s_xi = 4.0 → 13.0/5.0 = 2.6", 13.0, 4.0, 20.0);

        // ========== SMALL VALUES ==========
        run_test("F = 0.5, s_xi = 0.2 → 0.5/1.2 = 0.4166...", 0.5, 0.2, 20.0);
        run_test("F = 0.125, s_xi = 0.0 → 0.125/1.0 = 0.125", 0.125, 0.0, 20.0);
        run_test("F = 0.015625, s_xi = 0.0 → 0.015625/1.0", 0.015625, 0.0, 25.0);

        // ========== NEGATIVE NUMERATOR ==========
        run_test("F = -17.0, s_xi = 3.0 → -17.0/4.0 = -4.25", -17.0, 3.0, 20.0);
        run_test("F = -4.0, s_xi = 0.0 → -4.0/1.0 = -4.0", -4.0, 0.0, 20.0);
        run_test("F = -20.0, s_xi = 0.0 → -20.0/1.0 = -20.0", -20.0, 0.0, 20.0);

        // ========== EDGE CASES ==========
        run_test("F = 8.0, s_xi = 0.0 → 8.0/1.0 = 8.0", 8.0, 0.0, 20.0);
        run_test("F = 1.0, s_xi = 0.0 → 1.0/1.0 = 1.0", 1.0, 0.0, 5.0);
        run_test("F = 0.0, s_xi = 2.0 → 0.0/3.0 = 0.0", 0.0, 2.0, 20.0);

        // ========== LARGE VALUES ==========
        run_test("F = 20.0, s_xi = 9.0 → 20.0/10.0 = 2.0", 20.0, 9.0, 20.0);
        run_test("F = 30.0, s_xi = 0.5 → 30.0/1.5 = 20.0", 30.0, 0.5, 20.0);
        run_test("F = 31.0, s_xi = 30.0 → 31.0/31.0 = 1.0", 31.0, 30.0, 20.0);

        // ========== FRACTIONAL VALUES ==========
        run_test("F = 3.75, s_xi = 1.5 → 3.75/2.5 = 1.5", 3.75, 1.5, 20.0);
        run_test("F = 2.5, s_xi = 1.5 → 2.5/2.5 = 1.0", 2.5, 1.5, 5.0);
        run_test("F = 7.0, s_xi = 6.0 → 7.0/7.0 = 1.0", 7.0, 6.0, 5.0);

        // ========== NEGATIVE s_xi ==========
        run_test("F = 4.0, s_xi = -0.5 → 4.0/0.5 = 8.0", 4.0, -0.5, 20.0);
        run_test("F = -6.0, s_xi = -0.2 → -6.0/0.8 = -7.5", -6.0, -0.2, 20.0);
        run_test("F = -10.0, s_xi = -0.9 → -10.0/0.1 = -100.0", -10.0, -0.9, 30.0);

        // ========== POWER OF 2 (Mitchell baselines) ==========
        run_test("F = 16.0, s_xi = 0.0 → 16.0/1.0 = 16.0", 16.0, 0.0, 10.0);
        run_test("F = 8.0, s_xi = 1.0 → 8.0/2.0 = 4.0", 8.0, 1.0, 10.0);
        run_test("F = 32.0, s_xi = 3.0 → 32.0/4.0 = 8.0", 32.0, 3.0, 10.0);

        // ========== ACCURACY CHECKPOINTS (result ≈ 1.0) ==========
        run_test("F = 0.25, s_xi = 0.75 → 0.25/1.75 = 0.142857", 0.25, 0.75, 25.0);
        run_test("F = 0.03125, s_xi = 0.015625 → 0.03125/1.015625", 0.03125, 0.015625, 25.0);

        // ========== Q48.16 SPECIFIC TESTS - LARGE VALUES ==========
        run_test("F = 1000.0, s_xi = 999.0 → 1000.0/1000.0 = 1.0", 1000.0, 999.0, 20.0);
        run_test("F = 10000.0, s_xi = 0.0 → 10000.0/1.0 = 10000.0", 10000.0, 0.0, 20.0);
        run_test("F = 100.0, s_xi = 99.0 → 100.0/100.0 = 1.0", 100.0, 99.0, 20.0);

        // ========== EU OUTPUT RANGE TESTS ==========
        $display("\n╔════════════════════════════════════════════════════════════════════╗");
        $display("║  Testing with typical EU outputs (exp values from sigmoid)        ║");
        $display("╚════════════════════════════════════════════════════════════════════╝");
        
        run_test("F = 1.0, s_xi = 0.000045 (exp(-10)) → 1/1.000045", 1.0, 0.000045, 20.0);
        run_test("F = 1.0, s_xi = 1.0 (exp(0)) → 1/2", 1.0, 1.0, 10.0);
        run_test("F = 1.0, s_xi = 2.718 (exp(1)) → 1/3.718", 1.0, 2.718, 20.0);
        run_test("F = 1.0, s_xi = 148.413 (exp(5)) → 1/149.413", 1.0, 148.413, 20.0);
        run_test("F = 1.0, s_xi = 1000000.0 (exp(~14)) → 1/1000001", 1.0, 1000000.0, 30.0);

        // Final summary
        $display("\n╔════════════════════════════════════════════════════════════════════╗");
        $display("║                    TEST SUMMARY                                    ║");
        $display("╠════════════════════════════════════════════════════════════════════╣");
        $display("║  Total Tests:  %3d                                                 ║", test_count);
        $display("║  Passed:       %3d                                                 ║", pass_count);
        $display("║  Failed:       %3d                                                 ║", fail_count);
        $display("╠════════════════════════════════════════════════════════════════════╣");
        if (fail_count == 0) begin
            $display("║  ✅ ALL TESTS PASSED!                                             ║");
        end else begin
            $display("║  ❌ SOME TESTS FAILED                                             ║");
        end
        $display("╚════════════════════════════════════════════════════════════════════╝\n");

        $display("NOTE: Mitchell's algorithm has inherent ~5-20%% approximation error.");
        $display("      This is NORMAL and EXPECTED for this logarithmic approximation.\n");

        $finish;
    end

endmodule
