`timescale 1ns/1ps

//=============================================================================
// GCU Testbench - 32 Parallel GELU Units
// Tests 32 concurrent GELU computations with shared LUT
// Input range: [-6.4586, +5.6423] (typical BERT GELU range)
//=============================================================================
module GCU_tb;

    // =========================================================================
    // Parameters
    // =========================================================================
    localparam int Q = 16;              // Q48.16
    localparam int W = 64;              // 64-bit
    localparam int NUM_GELU = 32;       // 32 parallel GELU units
    localparam int NUM_LUT_PORTS = 64;  // 2 ports per GELU
    
    // Fixed-point conversion constant
    localparam real Q_SCALE = 2.0 ** Q;

    // =========================================================================
    // DUT Signals
    // =========================================================================
    logic signed [W-1:0] x [NUM_GELU-1:0];  // 32 inputs
    wire signed [W-1:0] y [NUM_GELU-1:0];   // 32 outputs

    // =========================================================================
    // Instantiate DUT (GCU with integrated SharedLUT)
    // =========================================================================
    GCU #(
        .Q(Q),
        .W(W),
        .NUM_GELU(NUM_GELU),
        .NUM_LUT_PORTS(NUM_LUT_PORTS)
    ) dut (
        .x(x),
        .y(y)
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
            return 64'sh8000000000000000;
        else
            return $rtoi(scaled);
    endfunction

    // Convert Q48.16 to real
    function automatic real fixed_to_real(logic signed [W-1:0] value);
        return $itor(value) / Q_SCALE;
    endfunction

    // Absolute value function
    function automatic real abs_real(real value);
        return (value < 0.0) ? -value : value;
    endfunction

    // =========================================================================
    // Golden Model 1: x / (1 + exp(-2*h(x)))
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
    // =========================================================================
    function automatic real gelu_golden_2(real x_val);
        real sqrt_2_over_pi;
        real x_cubed;
        real inner;
        real tanh_val;
        real exp_pos, exp_neg;
        
        sqrt_2_over_pi = 0.7978845608028654;
        x_cubed = x_val * x_val * x_val;
        inner = sqrt_2_over_pi * (x_val + 0.044715 * x_cubed);
        
        exp_pos = 2.71828182845904523536 ** inner;
        exp_neg = 2.71828182845904523536 ** (-inner);
        tanh_val = (exp_pos - exp_neg) / (exp_pos + exp_neg);
        
        return 0.5 * x_val * (1.0 + tanh_val);
    endfunction

    // =========================================================================
    // Test Statistics (per GELU unit)
    // =========================================================================
    int pass_count_g1 [NUM_GELU-1:0];
    int pass_count_g2 [NUM_GELU-1:0];
    int fail_count_g1 [NUM_GELU-1:0];
    int fail_count_g2 [NUM_GELU-1:0];
    int total_count [NUM_GELU-1:0];
    real max_error_g1 [NUM_GELU-1:0];
    real max_error_g2 [NUM_GELU-1:0];
    real sum_error_g1 [NUM_GELU-1:0];
    real sum_error_g2 [NUM_GELU-1:0];
    
    // Global statistics
    int global_pass_g1 = 0;
    int global_pass_g2 = 0;
    int global_fail_g1 = 0;
    int global_fail_g2 = 0;
    int global_total = 0;

    // Enable/disable detailed output per GELU
    parameter bit SHOW_PER_GELU_DETAILS = 0;  // Set to 1 for verbose output
    parameter bit SHOW_SUMMARY_ONLY = 1;       // Set to 1 for compact output

    // =========================================================================
    // Test Procedure
    // =========================================================================
    initial begin
        // Initialize statistics
        for (int i = 0; i < NUM_GELU; i++) begin
            pass_count_g1[i] = 0;
            pass_count_g2[i] = 0;
            fail_count_g1[i] = 0;
            fail_count_g2[i] = 0;
            total_count[i] = 0;
            max_error_g1[i] = 0.0;
            max_error_g2[i] = 0.0;
            sum_error_g1[i] = 0.0;
            sum_error_g2[i] = 0.0;
        end

        $display("\n╔════════════════════════════════════════════════════════════════════════════╗");
        $display("║                   GCU MODULE TESTBENCH - 32 Parallel GELUs                 ║");
        $display("║                   Flow: x[32] → 32×GELU → y[32]                            ║");
        $display("║                                                                            ║");
        $display("║  Format: Q48.16 (64-bit)                                                   ║");
        $display("║  Input Range:  [-6.4586, +5.6423] (BERT GELU typical range)               ║");
        $display("║  Parallelism: 32 independent GELU units                                    ║");
        $display("║  SharedLUT: 64 ports (2 per GELU)                                          ║");
        $display("║                                                                            ║");
        $display("║  Golden Model 1: x / (1 + exp(-2*1.702*x))                                ║");
        $display("║  Golden Model 2: 0.5*x*(1 + tanh(sqrt(2/π)*(x + 0.044715*x³)))            ║");
        $display("║                                                                            ║");
        $display("║  Tolerance: 30%% error OR absolute error < 0.05                            ║");
        $display("╚════════════════════════════════════════════════════════════════════════════╝\n");

        #10;
        $display("Starting test vectors...\n");

        // =====================================================================
        // Test Suite 1: All Same Value (test shared LUT under uniform load)
        // =====================================================================
        $display("════════════════════════════════════════════════════════════════════════════");
        $display("        TEST SUITE 1: UNIFORM INPUT (All GELUs same value)                 ");
        $display("════════════════════════════════════════════════════════════════════════════\n");
        
        run_parallel_test("All Zero", 0.0);
        run_parallel_test("All Unity", 1.0);
        run_parallel_test("All +2.5", 2.5);
        run_parallel_test("All -2.5", -2.5);

        // =====================================================================
        // Test Suite 2: Sequential Values (test all GELUs with different inputs)
        // =====================================================================
        $display("\n════════════════════════════════════════════════════════════════════════════");
        $display("        TEST SUITE 2: SEQUENTIAL INPUTS (Linearly spaced)                  ");
        $display("════════════════════════════════════════════════════════════════════════════\n");
        
        run_sequential_test("Linear -6 to +6", -6.0, 6.0);
        run_sequential_test("Linear -3 to +3", -3.0, 3.0);
        run_sequential_test("Linear 0 to +5", 0.0, 5.0);

        // =====================================================================
        // Test Suite 3: Random Values (stress test)
        // =====================================================================
        $display("\n════════════════════════════════════════════════════════════════════════════");
        $display("        TEST SUITE 3: RANDOM INPUTS (Monte Carlo)                          ");
        $display("════════════════════════════════════════════════════════════════════════════\n");
        
        for (int test = 0; test < 10; test++) begin
            run_random_test($sformatf("Random Test %0d", test+1), -6.0, 6.0);
        end

        // =====================================================================
        // Test Suite 4: Boundary Values (all GELUs)
        // =====================================================================
        $display("\n════════════════════════════════════════════════════════════════════════════");
        $display("        TEST SUITE 4: BOUNDARY VALUES                                      ");
        $display("════════════════════════════════════════════════════════════════════════════\n");
        
        run_parallel_test("Min boundary", -6.4586);
        run_parallel_test("Max boundary", 5.6423);
        run_parallel_test("Near zero +", 0.001);
        run_parallel_test("Near zero -", -0.001);

        // =====================================================================
        // Test Suite 5: Critical Transition Points
        // =====================================================================
        $display("\n════════════════════════════════════════════════════════════════════════════");
        $display("        TEST SUITE 5: CRITICAL TRANSITION POINTS                           ");
        $display("════════════════════════════════════════════════════════════════════════════\n");
        
        run_parallel_test("Transition +0.84", 0.8414);
        run_parallel_test("Transition -0.84", -0.8414);

        // =====================================================================
        // Test Suite 6: Alternating Pattern
        // =====================================================================
        $display("\n════════════════════════════════════════════════════════════════════════════");
        $display("        TEST SUITE 6: ALTERNATING PATTERNS                                 ");
        $display("════════════════════════════════════════════════════════════════════════════\n");
        
        run_alternating_test("Alternating ±2.5", 2.5, -2.5);
        run_alternating_test("Alternating ±1.0", 1.0, -1.0);

        // Display final summary
        $display("\n╔════════════════════════════════════════════════════════════════════════════╗");
        $display("║                         ALL TESTS COMPLETED                                ║");
        $display("║                                                                            ║");
        display_global_summary();
        $display("╚════════════════════════════════════════════════════════════════════════════╝\n");
        
        $finish;
    end

    // =========================================================================
    // Test Tasks
    // =========================================================================

    // Test all GELUs with same input value
    task automatic run_parallel_test(string test_name, real x_val);
        real x_vals [NUM_GELU-1:0];
        
        // Set all inputs to same value
        for (int i = 0; i < NUM_GELU; i++) begin
            x_vals[i] = x_val;
        end
        
        run_test_vector(test_name, x_vals);
    endtask

    // Test GELUs with sequential values (linearly spaced)
    task automatic run_sequential_test(string test_name, real start_val, real end_val);
        real x_vals [NUM_GELU-1:0];
        real step;
        
        step = (end_val - start_val) / real'(NUM_GELU - 1);
        
        for (int i = 0; i < NUM_GELU; i++) begin
            x_vals[i] = start_val + (step * real'(i));
        end
        
        run_test_vector(test_name, x_vals);
    endtask

    // Test GELUs with random values
    task automatic run_random_test(string test_name, real min_val, real max_val);
        real x_vals [NUM_GELU-1:0];
        real range;
        int seed;
        
        seed = $urandom();
        range = max_val - min_val;
        
        for (int i = 0; i < NUM_GELU; i++) begin
            x_vals[i] = min_val + (range * ($urandom() / 4294967296.0));
        end
        
        run_test_vector(test_name, x_vals);
    endtask

    // Test GELUs with alternating pattern
    task automatic run_alternating_test(string test_name, real val1, real val2);
        real x_vals [NUM_GELU-1:0];
        
        for (int i = 0; i < NUM_GELU; i++) begin
            x_vals[i] = (i % 2 == 0) ? val1 : val2;
        end
        
        run_test_vector(test_name, x_vals);
    endtask

    // Core test execution
    task automatic run_test_vector(string test_name, real x_vals [NUM_GELU-1:0]);
        real y_golden_1 [NUM_GELU-1:0];
        real y_golden_2 [NUM_GELU-1:0];
        real y_actual [NUM_GELU-1:0];
        real error_abs_g1 [NUM_GELU-1:0];
        real error_abs_g2 [NUM_GELU-1:0];
        real error_pct_g1 [NUM_GELU-1:0];
        real error_pct_g2 [NUM_GELU-1:0];
        int pass_g1, pass_g2, fail_g1, fail_g2;
        
        pass_g1 = 0;
        pass_g2 = 0;
        fail_g1 = 0;
        fail_g2 = 0;
        
        // Set inputs
        for (int i = 0; i < NUM_GELU; i++) begin
            x[i] = real_to_fixed(x_vals[i]);
        end
        
        // Wait for combinational propagation
        #100;
        
        // Compute golden models and actual results for all GELUs
        for (int i = 0; i < NUM_GELU; i++) begin
            y_golden_1[i] = gelu_golden_1(x_vals[i]);
            y_golden_2[i] = gelu_golden_2(x_vals[i]);
            y_actual[i] = fixed_to_real(y[i]);
            
            // Compute errors for Golden Model 1
            error_abs_g1[i] = y_actual[i] - y_golden_1[i];
            if (y_golden_1[i] != 0.0)
                error_pct_g1[i] = (error_abs_g1[i] / y_golden_1[i]) * 100.0;
            else
                error_pct_g1[i] = (y_actual[i] != 0.0) ? 100.0 : 0.0;
            
            // Compute errors for Golden Model 2
            error_abs_g2[i] = y_actual[i] - y_golden_2[i];
            if (y_golden_2[i] != 0.0)
                error_pct_g2[i] = (error_abs_g2[i] / y_golden_2[i]) * 100.0;
            else
                error_pct_g2[i] = (y_actual[i] != 0.0) ? 100.0 : 0.0;
            
            // Update statistics
            total_count[i]++;
            sum_error_g1[i] += abs_real(error_pct_g1[i]);
            sum_error_g2[i] += abs_real(error_pct_g2[i]);
            
            if (abs_real(error_pct_g1[i]) > max_error_g1[i])
                max_error_g1[i] = abs_real(error_pct_g1[i]);
            
            if (abs_real(error_pct_g2[i]) > max_error_g2[i])
                max_error_g2[i] = abs_real(error_pct_g2[i]);
            
            // Determine pass/fail
            if (abs_real(error_pct_g1[i]) < 30.0 || abs_real(error_abs_g1[i]) < 0.05) begin
                pass_count_g1[i]++;
                pass_g1++;
            end else begin
                fail_count_g1[i]++;
                fail_g1++;
            end
            
            if (abs_real(error_pct_g2[i]) < 30.0 || abs_real(error_abs_g2[i]) < 0.05) begin
                pass_count_g2[i]++;
                pass_g2++;
            end else begin
                fail_count_g2[i]++;
                fail_g2++;
            end
        end
        
        // Update global statistics
        global_pass_g1 += pass_g1;
        global_pass_g2 += pass_g2;
        global_fail_g1 += fail_g1;
        global_fail_g2 += fail_g2;
        global_total += NUM_GELU;
        
        // Display results
        if (!SHOW_SUMMARY_ONLY) begin
            $display("────────────────────────────────────────────────────────────────────────────");
            $display("TEST: %s", test_name);
            $display("────────────────────────────────────────────────────────────────────────────");
            
            if (SHOW_PER_GELU_DETAILS) begin
                // Detailed per-GELU output
                for (int i = 0; i < NUM_GELU; i++) begin
                    $display("  GELU[%2d]: x=%8.4f → y=%8.4f | Golden1=%8.4f (err=%6.2f%%) | Golden2=%8.4f (err=%6.2f%%)",
                             i, x_vals[i], y_actual[i], 
                             y_golden_1[i], error_pct_g1[i],
                             y_golden_2[i], error_pct_g2[i]);
                end
            end
            
            $display("  Golden Model 1: %2d/%2d PASS, %2d/%2d FAIL", pass_g1, NUM_GELU, fail_g1, NUM_GELU);
            $display("  Golden Model 2: %2d/%2d PASS, %2d/%2d FAIL", pass_g2, NUM_GELU, fail_g2, NUM_GELU);
            $display("");
        end else begin
            // Compact summary
            string status1 = (fail_g1 == 0) ? "✓ PASS" : "✗ FAIL";
            string status2 = (fail_g2 == 0) ? "✓ PASS" : "✗ FAIL";
            $display("%-40s | G1: %2d/%2d %s | G2: %2d/%2d %s", 
                     test_name, pass_g1, NUM_GELU, status1, pass_g2, NUM_GELU, status2);
        end
        
        #10;
    endtask

    // =========================================================================
    // Summary Display
    // =========================================================================
    task display_global_summary();
        real pass_rate_g1, pass_rate_g2;
        real avg_error_g1, avg_error_g2;
        real total_avg_error_g1, total_avg_error_g2;
        
        pass_rate_g1 = (real'(global_pass_g1) / real'(global_total)) * 100.0;
        pass_rate_g2 = (real'(global_pass_g2) / real'(global_total)) * 100.0;
        
        // Calculate average error across all GELUs
        total_avg_error_g1 = 0.0;
        total_avg_error_g2 = 0.0;
        for (int i = 0; i < NUM_GELU; i++) begin
            if (total_count[i] > 0) begin
                total_avg_error_g1 += sum_error_g1[i] / real'(total_count[i]);
                total_avg_error_g2 += sum_error_g2[i] / real'(total_count[i]);
            end
        end
        total_avg_error_g1 /= real'(NUM_GELU);
        total_avg_error_g2 /= real'(NUM_GELU);
        
        $display("║                          GLOBAL TEST SUMMARY                               ║");
        $display("║                                                                            ║");
        $display("║ Total Tests:       %6d (across %2d parallel GELU units)                  ║", 
                 global_total, NUM_GELU);
        $display("║ Tests per GELU:    %6d                                                    ║", 
                 total_count[0]);
        $display("║                                                                            ║");
        $display("║ ══════════════════════════════════════════════════════════════════════════ ║");
        $display("║ Golden Model 1: x / (1 + exp(-2*1.702*x))                                 ║");
        $display("║ ══════════════════════════════════════════════════════════════════════════ ║");
        $display("║   Total Passed:    %6d / %6d                                             ║", 
                 global_pass_g1, global_total);
        $display("║   Total Failed:    %6d / %6d                                             ║", 
                 global_fail_g1, global_total);
        $display("║   Pass Rate:       %6.2f%%                                                 ║", 
                 pass_rate_g1);
        $display("║   Avg Error:       %6.2f%% (across all GELUs)                              ║", 
                 total_avg_error_g1);
        $display("║                                                                            ║");
        $display("║ ══════════════════════════════════════════════════════════════════════════ ║");
        $display("║ Golden Model 2: 0.5*x*(1 + tanh(sqrt(2/π)*(x + 0.044715*x³)))             ║");
        $display("║ ══════════════════════════════════════════════════════════════════════════ ║");
        $display("║   Total Passed:    %6d / %6d                                             ║", 
                 global_pass_g2, global_total);
        $display("║   Total Failed:    %6d / %6d                                             ║", 
                 global_fail_g2, global_total);
        $display("║   Pass Rate:       %6.2f%%                                                 ║", 
                 pass_rate_g2);
        $display("║   Avg Error:       %6.2f%% (across all GELUs)                              ║", 
                 total_avg_error_g2);
        $display("║                                                                            ║");
        
        if (global_fail_g1 == 0 && global_fail_g2 == 0) begin
            $display("║ ✅ ALL TESTS PASSED! 32-parallel GCU implementation verified.             ║");
        end else begin
            $display("║ ⚠️  Some tests failed. Review errors above for analysis.                 ║");
        end
        
        // Display per-GELU statistics
        $display("║                                                                            ║");
        $display("║ ══════════════════════════════════════════════════════════════════════════ ║");
        $display("║                       PER-GELU STATISTICS                                  ║");
        $display("║ ══════════════════════════════════════════════════════════════════════════ ║");
        
        for (int i = 0; i < NUM_GELU; i++) begin
            if (total_count[i] > 0) begin
                avg_error_g1 = sum_error_g1[i] / real'(total_count[i]);
                avg_error_g2 = sum_error_g2[i] / real'(total_count[i]);
                $display("║ GELU[%2d]: G1=%3d/%3d (Avg:%5.2f%%) | G2=%3d/%3d (Avg:%5.2f%%)            ║",
                         i, 
                         pass_count_g1[i], total_count[i], avg_error_g1,
                         pass_count_g2[i], total_count[i], avg_error_g2);
            end
        end
    endtask

endmodule
