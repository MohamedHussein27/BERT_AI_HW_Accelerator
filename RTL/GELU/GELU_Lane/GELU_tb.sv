`timescale 1ns/1ps

module GELU_tb;

    // Parameters
    localparam int Q = 16;    // Q48.16
    localparam int W = 64;    // 64-bit
    
    // Fixed-point conversion constant
    localparam real Q_SCALE = 2.0 ** Q;

    // DUT signals
    logic signed [W-1:0] x;
    wire signed [W-1:0] y;
    
    // LUT interface signals - 32-bit from SharedLUT
    wire [2:0] segment_index_0;
    wire [2:0] segment_index_1;
    wire signed [31:0] k_coeff_0_32;      // 32-bit from LUT
    wire signed [31:0] b_intercept_0_32;  // 32-bit from LUT
    wire signed [31:0] k_coeff_1_32;      // 32-bit from LUT
    wire signed [31:0] b_intercept_1_32;  // 32-bit from LUT
    
    // Sign-extend 32-bit LUT outputs to 64-bit for GELU interface
    wire signed [W-1:0] k_coeff_0;
    wire signed [W-1:0] b_intercept_0;
    wire signed [W-1:0] k_coeff_1;
    wire signed [W-1:0] b_intercept_1;
    
    assign k_coeff_0 = {{32{k_coeff_0_32[31]}}, k_coeff_0_32};
    assign b_intercept_0 = {{32{b_intercept_0_32[31]}}, b_intercept_0_32};
    assign k_coeff_1 = {{32{k_coeff_1_32[31]}}, k_coeff_1_32};
    assign b_intercept_1 = {{32{b_intercept_1_32[31]}}, b_intercept_1_32};

    // =========================================================================
    // Internal Signal Probes for Debugging
    // =========================================================================
    wire signed [W-1:0] s_x;           // Polynomial output
    wire signed [W-1:0] exp_s_x;       // EU1 output (exp(s_x))
    wire signed [W-1:0] du_exponent;   // DU exponent output
    wire du_sign;                       // DU sign output
    wire signed [W-1:0] eu2_result;    // EU2 output (before sign)

    assign s_x = dut.s_x_q4816;
    assign exp_s_x = dut.exp_s_x;
    assign du_exponent = dut.du_exponent;
    assign du_sign = dut.du_sign;
    assign eu2_result = dut.eu2_result;

    // Instantiate SharedLUT (2 ports for single GELU)
    wire [2:0] segment_indices [1:0];
    wire signed [31:0] k_coeffs [1:0];      // 32-bit arrays
    wire signed [31:0] b_intercepts [1:0];  // 32-bit arrays
    
    assign segment_indices[0] = segment_index_0;
    assign segment_indices[1] = segment_index_1;
    assign k_coeff_0_32 = k_coeffs[0];
    assign b_intercept_0_32 = b_intercepts[0];
    assign k_coeff_1_32 = k_coeffs[1];
    assign b_intercept_1_32 = b_intercepts[1];
    
    SharedLUT #(
        .Q(22),
        .W(32),
        .NUM_SEGMENTS(8),
        .NUM_PORTS(2)
    ) lut_inst (
        .segment_index(segment_indices),
        .k_coeff(k_coeffs),
        .b_intercept(b_intercepts)
    );

    // Instantiate DUT
    GELU #(
        .Q(Q),
        .W(W),
        .LUT_PORT_BASE(0)
    ) dut (
        .x(x),
        .y(y),
        .segment_index_0(segment_index_0),
        .segment_index_1(segment_index_1),
        .k_coeff_0(k_coeff_0),           // 64-bit to GELU
        .b_intercept_0(b_intercept_0),   // 64-bit to GELU
        .k_coeff_1(k_coeff_1),           // 64-bit to GELU
        .b_intercept_1(b_intercept_1)    // 64-bit to GELU
    );

    // =========================================================================
    // Helper Functions
    // =========================================================================
    
    // Convert real to Q48.16
    function automatic logic signed [W-1:0] real_to_fixed(real value);
        real scaled;
        scaled = value * Q_SCALE;
        if (scaled > 9223372036854775807.0)
            return 64'sh7FFFFFFFFFFFFFFF;
        else if (scaled < -9223372036854775808.0)
            return -64'sh8000000000000000;
        else
            return $rtoi(scaled);
    endfunction

    // Convert Q48.16 to real
    function automatic real fixed_to_real(logic signed [W-1:0] value);
        return $itor(value) / Q_SCALE;
    endfunction

    // Absolute value function (replacement for $abs)
    function automatic real abs_real(real value);
        return (value < 0.0) ? -value : value;
    endfunction

    // =========================================================================
    // Golden Model 1: x / (1 + exp(-2*h(x)))
    // Matches hardware flow: h(x) ≈ 1.702*x polynomial approximation
    // =========================================================================
    function automatic real gelu_golden_1(real x_val);
        real h_x;
        real exp_term;
        
        h_x = 1.702 * x_val;
        exp_term = 2.71828182845904523536 ** (-2.0 * h_x);
        
        return x_val / (1.0 + exp_term);
    endfunction

    // =========================================================================
    // Golden Model 2: Standard GELU
    // g(x) = 0.5*x*(1 + tanh(sqrt(2/π)*(x + 0.044715*x³)))
    // Most accurate reference (used in PyTorch/TensorFlow)
    // =========================================================================
    function automatic real gelu_golden_2(real x_val);
        real sqrt_2_over_pi;
        real x_cubed;
        real inner;
        real tanh_val;
        real exp_pos, exp_neg;
        
        sqrt_2_over_pi = 0.7978845608028654;  // sqrt(2/π)
        x_cubed = x_val * x_val * x_val;
        inner = sqrt_2_over_pi * (x_val + 0.044715 * x_cubed);
        
        // Compute tanh(inner) = (e^inner - e^(-inner)) / (e^inner + e^(-inner))
        exp_pos = 2.71828182845904523536 ** inner;
        exp_neg = 2.71828182845904523536 ** (-inner);
        tanh_val = (exp_pos - exp_neg) / (exp_pos + exp_neg);
        
        return 0.5 * x_val * (1.0 + tanh_val);
    endfunction

    // =========================================================================
    // Test Procedure
    // =========================================================================
    initial begin
        $display("\n╔════════════════════════════════════════════════════════════════════════════╗");
        $display("║                      GELU MODULE TESTBENCH - Q48.16                        ║");
        $display("║                   Flow: x → Poly → EU → DU → EU → y                       ║");
        $display("║                                                                            ║");
        $display("║  Input Range:  [-6.4586, +5.6423] (BERT GELU typical range)               ║");
        $display("║                                                                            ║");
        $display("║  Architecture:                                                             ║");
        $display("║    1. PolynomialUnit: s(x) = -2.3125*(x + 0.046875*x³)                    ║");
        $display("║    2. EU1: exp(s(x)) = 2^s(x) using LUT-based approximation               ║");
        $display("║    3. DU: log2(x / (1 + exp(s(x)))) using Mitchell's algorithm            ║");
        $display("║    4. EU2: 2^exponent antilog operation                                    ║");
        $display("║                                                                            ║");
        $display("║  Golden Model 1: x / (1 + exp(-2*1.702*x))                                ║");
        $display("║  Golden Model 2: 0.5*x*(1 + tanh(sqrt(2/π)*(x + 0.044715*x³)))            ║");
        $display("║                                                                            ║");
        $display("║  Tolerance: 30%% error OR absolute error < 0.05                            ║");
        $display("╚════════════════════════════════════════════════════════════════════════════╝\n");

        #10;
        $display("Starting test vectors...\n");

        // =====================================================================
        // Test Suite 1: BOUNDARY VALUES
        // =====================================================================
        $display("════════════════════════════════════════════════════════════════════════════");
        $display("                           BOUNDARY VALUES                                  ");
        $display("════════════════════════════════════════════════════════════════════════════\n");
        run_test("Min boundary", -6.4586);
        run_test("Max boundary", 5.6423);
        run_test("Zero", 0.0);

        // =====================================================================
        // Test Suite 2: POSITIVE VALUES (0 to +5.6423)
        // =====================================================================
        $display("\n════════════════════════════════════════════════════════════════════════════");
        $display("                       POSITIVE VALUES [0, 5.6423]                          ");
        $display("════════════════════════════════════════════════════════════════════════════\n");
        run_test("Very small pos", 0.01);
        run_test("Tiny pos", 0.1);
        run_test("Quarter", 0.25);
        run_test("Half", 0.5);
        run_test("Three quarters", 0.75);
        run_test("Unity", 1.0);
        run_test("1.25", 1.25);
        run_test("1.5", 1.5);
        run_test("1.75", 1.75);
        run_test("Two", 2.0);
        run_test("2.5", 2.5);
        run_test("Three", 3.0);
        run_test("3.5", 3.5);
        run_test("Four", 4.0);
        run_test("4.5", 4.5);
        run_test("Five", 5.0);
        run_test("5.5", 5.5);
        run_test("Near max", 5.6);
        
        // =====================================================================
        // Test Suite 3: NEGATIVE VALUES (-6.4586 to 0)
        // =====================================================================
        $display("\n════════════════════════════════════════════════════════════════════════════");
        $display("                      NEGATIVE VALUES [-6.4586, 0]                          ");
        $display("════════════════════════════════════════════════════════════════════════════\n");
        run_test("Very small neg", -0.01);
        run_test("Tiny neg", -0.1);
        run_test("Minus quarter", -0.25);
        run_test("Minus half", -0.5);
        run_test("Minus three quarters", -0.75);
        run_test("Minus one", -1.0);
        run_test("-1.25", -1.25);
        run_test("-1.5", -1.5);
        run_test("-1.75", -1.75);
        run_test("Minus two", -2.0);
        run_test("-2.5", -2.5);
        run_test("Minus three", -3.0);
        run_test("-3.5", -3.5);
        run_test("Minus four", -4.0);
        run_test("-4.5", -4.5);
        run_test("Minus five", -5.0);
        run_test("-5.5", -5.5);
        run_test("Minus six", -6.0);
        run_test("Near min", -6.4);
        
        // =====================================================================
        // Test Suite 4: CRITICAL TRANSITION POINTS
        // =====================================================================
        $display("\n════════════════════════════════════════════════════════════════════════════");
        $display("                      CRITICAL TRANSITION POINTS                            ");
        $display("════════════════════════════════════════════════════════════════════════════\n");
        run_test("Transition +0.84", 0.8414);
        run_test("Transition -0.84", -0.8414);
        run_test("99% saturation +", 2.5);
        run_test("99% saturation -", -2.5);
        run_test("Linear region +", 0.3);
        run_test("Linear region -", -0.3);
        
        // =====================================================================
        // Test Suite 5: DENSE SWEEP (for accuracy profiling)
        // =====================================================================
        $display("\n════════════════════════════════════════════════════════════════════════════");
        $display("                    DENSE SWEEP [-6 to +6, step 0.5]                       ");
        $display("════════════════════════════════════════════════════════════════════════════\n");
        for (real val = -6.0; val <= 6.0; val += 0.5) begin
            if (val >= -6.4586 && val <= 5.6423) begin
                run_test($sformatf("Sweep %.1f", val), val);
            end
        end

        // =====================================================================
        // Test Suite 6: OUT-OF-RANGE VALUES (for saturation testing)
        // =====================================================================
        $display("\n════════════════════════════════════════════════════════════════════════════");
        $display("                   OUT-OF-RANGE (Saturation Test)                          ");
        $display("════════════════════════════════════════════════════════════════════════════\n");
        run_test("Beyond max +7", 7.0);
        run_test("Beyond max +10", 10.0);
        run_test("Beyond min -7", -7.0);
        run_test("Beyond min -10", -10.0);

        $display("\n╔════════════════════════════════════════════════════════════════════════════╗");
        $display("║                         ALL TESTS COMPLETED                                ║");
        $display("║                                                                            ║");
        display_summary();
        $display("╚════════════════════════════════════════════════════════════════════════════╝\n");
        
        $finish;
    end

    // Test statistics
    int pass_count_g1 = 0;
    int pass_count_g2 = 0;
    int fail_count_g1 = 0;
    int fail_count_g2 = 0;
    int total_count = 0;
    real max_error_g1 = 0.0;
    real max_error_g2 = 0.0;
    real sum_error_g1 = 0.0;
    real sum_error_g2 = 0.0;
    string max_error_test_g1 = "";
    string max_error_test_g2 = "";

    // Enable/disable detailed internal signal display
    parameter bit SHOW_INTERNALS = 1;

    // Test execution task
    task automatic run_test(string test_name, real x_val);
        real y_golden_1, y_golden_2;
        real y_actual;
        real error_abs_g1, error_abs_g2;
        real error_pct_g1, error_pct_g2;
        string status_g1, status_g2;
        
        // Set input
        x = real_to_fixed(x_val);
        
        // Wait for combinational propagation
        #50;
        
        // Compute golden models
        y_golden_1 = gelu_golden_1(x_val);
        y_golden_2 = gelu_golden_2(x_val);
        
        // Get actual result
        y_actual = fixed_to_real(y);
        
        // Compute errors for Golden Model 1
        error_abs_g1 = y_actual - y_golden_1;
        if (y_golden_1 != 0.0)
            error_pct_g1 = (error_abs_g1 / y_golden_1) * 100.0;
        else if (y_actual != 0.0)
            error_pct_g1 = 100.0;
        else
            error_pct_g1 = 0.0;
        
        // Compute errors for Golden Model 2
        error_abs_g2 = y_actual - y_golden_2;
        if (y_golden_2 != 0.0)
            error_pct_g2 = (error_abs_g2 / y_golden_2) * 100.0;
        else if (y_actual != 0.0)
            error_pct_g2 = 100.0;
        else
            error_pct_g2 = 0.0;
        
        // Track statistics using abs_real function instead of $abs
        total_count++;
        sum_error_g1 += abs_real(error_pct_g1);
        sum_error_g2 += abs_real(error_pct_g2);
        
        if (abs_real(error_pct_g1) > max_error_g1) begin
            max_error_g1 = abs_real(error_pct_g1);
            max_error_test_g1 = test_name;
        end
        
        if (abs_real(error_pct_g2) > max_error_g2) begin
            max_error_g2 = abs_real(error_pct_g2);
            max_error_test_g2 = test_name;
        end
        
        // Determine pass/fail (30% tolerance for Mitchell approximation)
        if (abs_real(error_pct_g1) < 30.0 || abs_real(error_abs_g1) < 0.05) begin
            status_g1 = "✓ PASS";
            pass_count_g1++;
        end else begin
            status_g1 = "✗ FAIL";
            fail_count_g1++;
        end
        
        if (abs_real(error_pct_g2) < 30.0 || abs_real(error_abs_g2) < 0.05) begin
            status_g2 = "✓ PASS";
            pass_count_g2++;
        end else begin
            status_g2 = "✗ FAIL";
            fail_count_g2++;
        end
        
        // Display results
        $display("╔════════════════════════════════════════════════════════════════════════════╗");
        $display("║ TEST #%-2d: %-64s ║", total_count, test_name);
        $display("╠════════════════════════════════════════════════════════════════════════════╣");
        $display("║ Input (x):           %12.6f  (0x%016h)                 ║", x_val, x);
        
        // Show internal pipeline stages if enabled
        if (SHOW_INTERNALS) begin
            $display("║                                                                            ║");
            $display("║ INTERNAL PIPELINE:                                                         ║");
            $display("║   Poly s(x):       %12.6f  (0x%016h)                 ║", 
                     fixed_to_real(s_x), s_x);
            $display("║   EU1 exp(s):      %12.6f  (0x%016h)                 ║", 
                     fixed_to_real(exp_s_x), exp_s_x);
            $display("║   DU exponent:     %12.6f  (0x%016h)                 ║", 
                     fixed_to_real(du_exponent), du_exponent);
            $display("║   DU sign:         %b (%s)                                              ║", 
                     du_sign, du_sign ? "NEG" : "POS");
            $display("║   EU2 result:      %12.6f  (0x%016h)                 ║", 
                     fixed_to_real(eu2_result), eu2_result);
            $display("║   LUT indices:     EU1=%0d, EU2=%0d                                          ║",
                     segment_index_0, segment_index_1);
        end
        
        $display("║                                                                            ║");
        $display("║ Golden Model 1:      %12.6f                                      ║", y_golden_1);
        $display("║   Error (abs):       %12.6f                                      ║", error_abs_g1);
        $display("║   Error (%%):         %12.6f%%                                     ║", error_pct_g1);
        $display("║   Status:            %-52s ║", status_g1);
        $display("║                                                                            ║");
        $display("║ Golden Model 2:      %12.6f                                      ║", y_golden_2);
        $display("║   Error (abs):       %12.6f                                      ║", error_abs_g2);
        $display("║   Error (%%):         %12.6f%%                                     ║", error_pct_g2);
        $display("║   Status:            %-52s ║", status_g2);
        $display("║                                                                            ║");
        $display("║ Actual Output (y):   %12.6f  (0x%016h)                 ║", y_actual, y);
        $display("╚════════════════════════════════════════════════════════════════════════════╝\n");
        
        #10;
    endtask

    // Display test summary
    task display_summary();
        real pass_rate_g1, pass_rate_g2;
        real avg_error_g1, avg_error_g2;
        
        pass_rate_g1 = (real'(pass_count_g1) / real'(total_count)) * 100.0;
        pass_rate_g2 = (real'(pass_count_g2) / real'(total_count)) * 100.0;
        avg_error_g1 = sum_error_g1 / real'(total_count);
        avg_error_g2 = sum_error_g2 / real'(total_count);
        
        $display("║                           TEST SUMMARY                                     ║");
        $display("║                                                                            ║");
        $display("║ Total Tests:     %3d                                                       ║", total_count);
        $display("║ Input Range:     [-6.4586, +5.6423] (BERT GELU range)                     ║");
        $display("║                                                                            ║");
        $display("║ ══════════════════════════════════════════════════════════════════════════ ║");
        $display("║ Golden Model 1: x / (1 + exp(-2*1.702*x))                                 ║");
        $display("║ ══════════════════════════════════════════════════════════════════════════ ║");
        $display("║   Passed:        %3d / %3d                                                 ║", pass_count_g1, total_count);
        $display("║   Failed:        %3d / %3d                                                 ║", fail_count_g1, total_count);
        $display("║   Pass Rate:     %6.2f%%                                                   ║", pass_rate_g1);
        $display("║   Max Error:     %6.2f%% (test: %-30s)            ║", max_error_g1, max_error_test_g1);
        $display("║   Avg Error:     %6.2f%%                                                   ║", avg_error_g1);
        $display("║                                                                            ║");
        $display("║ ══════════════════════════════════════════════════════════════════════════ ║");
        $display("║ Golden Model 2: 0.5*x*(1 + tanh(sqrt(2/π)*(x + 0.044715*x³)))             ║");
        $display("║ ══════════════════════════════════════════════════════════════════════════ ║");
        $display("║   Passed:        %3d / %3d                                                 ║", pass_count_g2, total_count);
        $display("║   Failed:        %3d / %3d                                                 ║", fail_count_g2, total_count);
        $display("║   Pass Rate:     %6.2f%%                                                   ║", pass_rate_g2);
        $display("║   Max Error:     %6.2f%% (test: %-30s)            ║", max_error_g2, max_error_test_g2);
        $display("║   Avg Error:     %6.2f%%                                                   ║", avg_error_g2);
        $display("║                                                                            ║");
        
        if (fail_count_g1 == 0 && fail_count_g2 == 0) begin
            $display("║ ✅ ALL TESTS PASSED! Hardware GELU implementation verified.              ║");
        end else begin
            $display("║ ⚠️  Some tests failed. Review errors above for analysis.                 ║");
        end
    endtask

endmodule
