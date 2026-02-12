`timescale 1ns/1ps

module PolynomialUnit_tb;

    parameter WIDTH = 32;
    parameter Q = 22;  // Q10.22 format (1 sign + 9 integer + 22 fractional)
    
    // DUT signals
    reg  signed [WIDTH-1:0] x;
    wire signed [WIDTH-1:0] s_x;
    
    // Instantiate DUT
    PolynomialUnit #(
        .WIDTH(WIDTH),
        .Q(Q)
    ) dut (
        .x(x),
        .s_x(s_x)
    );
    
    // Convert Q10.22 integer to real
    function real q1022_to_real;
        input signed [WIDTH-1:0] val;
        begin
            q1022_to_real = $itor(val) / (2.0 ** Q);
        end
    endfunction
    
    // Convert real to Q10.22 integer
    function signed [WIDTH-1:0] real_to_q1022;
        input real val;
        real scaled;
        begin
            scaled = val * (2.0 ** Q);
            if (scaled > 2147483647.0)
                real_to_q1022 = 32'sd2147483647;
            else if (scaled < -2147483648.0)
                real_to_q1022 = -32'sd2147483648;
            else
                real_to_q1022 = $rtoi(scaled);
        end
    endfunction
    
    // ACTUAL EXPECTED: s(x) = -2*log2(e) * sqrt(2/π) * (x + 0.044715*x³)
    function real compute_expected;
        input real x_val;
        real log2_e, sqrt_2_over_pi, x_cubed, inner, result;
        begin
            log2_e = 1.4426950408889634;          // log2(e)
            sqrt_2_over_pi = 0.7978845608028654;  // sqrt(2/π)
            
            x_cubed = x_val * x_val * x_val;
            inner = x_val + 0.044715 * x_cubed;
            result = -2.0 * log2_e * sqrt_2_over_pi * inner;
            
            compute_expected = result;
        end
    endfunction
    
    // HARDWARE APPROXIMATION: h(x) = -K1*(x + K2*x³)
    // where K1 ≈ 2*log2(e)*sqrt(2/π) = 2.3026, K2 ≈ 0.044715
    function real compute_hw_approx;
        input real x_val;
        real K1, K2, x_cubed, inner;
        begin
            K1 = 2.3125;      // Hardware approximation: 2 + 1/4 + 1/16
            K2 = 0.046875;    // Hardware approximation: 1/32 + 1/64 = 3/64
            
            x_cubed = x_val * x_val * x_val;
            inner = x_val + K2 * x_cubed;
            compute_hw_approx = -K1 * inner;
        end
    endfunction
    
    // Test procedure    
    real x_real, s_x_real, expected_exact, expected_hw;
    real abs_error_exact, abs_error_hw, rel_error_exact, rel_error_hw;
    integer i, pass_count, fail_count, total_tests;
    real max_abs_error, max_rel_error, max_hw_error;
    real worst_x, worst_rel_x, worst_hw_x;
    
    initial begin
        pass_count = 0;
        fail_count = 0;
        max_abs_error = 0.0;
        max_rel_error = 0.0;
        max_hw_error = 0.0;
        worst_x = 0.0;
        worst_rel_x = 0.0;
        worst_hw_x = 0.0;
        
        // Display header
        $display("====================================================================================");
        $display("              POLYNOMIAL UNIT TESTBENCH - Q10.22 FORMAT                            ");
        $display("====================================================================================");
        $display("Configuration:");
        $display("  WIDTH: %0d bits", WIDTH);
        $display("  Q Format: Q10.%0d (1 sign + 9 integer bits, 22 fractional bits)", Q);
        $display("  Q10.22 Range: [-512, 511.999756]");
        $display("  Input Range: [-6.4586, +5.6423]");
        $display("");
        $display("Expected Formula (EXACT):");
        $display("  s(x) = -2*log₂(e) * √(2/π) * (x + 0.044715*x³)");
        $display("       = -2.302585 * (x + 0.044715*x³)");
        $display("");
        $display("Hardware Implementation (APPROXIMATION):");
        $display("  h(x) = -K1 * (x + K2*x³)");
        $display("    K1 = 2.3125   (2 + 1/4 + 1/16)           ≈ 2.302585");
        $display("    K2 = 0.046875 (1/32 + 1/64 = 3/64)       ≈ 0.044715");
        $display("");
        
        // Test 1: Boundary values
        $display("====================================================================================");
        $display("                    TEST 1: BOUNDARY VALUES                                        ");
        $display("====================================================================================");
        $display("  x_real  |  HW Output | Expected  | Err(Exact) | Err(vs HW) | Rel Err%%  | Status  ");
        $display("------------------------------------------------------------------------------------");
        
        // Test exact boundaries
        test_value(-6.4586);
        test_value(5.6423);
        test_value(0.0);
        test_value(-6.0);
        test_value(5.0);
        test_value(-3.0);
        test_value(3.0);
        test_value(-1.0);
        test_value(1.0);
                
        // Test 2: Comprehensive sweep
        $display("====================================================================================");
        $display("           TEST 2: COMPREHENSIVE RANGE SWEEP (step=0.1)                           ");
        $display("====================================================================================");
        $display("  x_real  |  HW Output | Expected  | Err(Exact) | Err(vs HW) | Rel Err%%  | Status  ");
        $display("------------------------------------------------------------------------------------");
        
        // Sweep from -6.5 to +5.7 in steps of 0.1
        for (i = -65; i <= 57; i = i + 1) begin
            x_real = $itor(i) / 10.0;
            
            // Skip if outside measured range
            if (x_real < -6.4586 || x_real > 5.6423)
                continue;
                
            test_value(x_real);
        end
                
        // Test 3: Fine resolution around critical points
        $display("====================================================================================");
        $display("       TEST 3: FINE RESOLUTION AROUND EXTREMES (step=0.01)                        ");
        $display("====================================================================================");
        
        // Around minimum (-6.4586)
        for (i = -6459; i <= -6400; i = i + 5) begin
            x_real = $itor(i) / 1000.0;
            test_value(x_real);
        end
        
        // Around maximum (5.6423)
        for (i = 5600; i <= 5643; i = i + 5) begin
            x_real = $itor(i) / 1000.0;
            test_value(x_real);
        end
        
        // Around zero
        for (i = -50; i <= 50; i = i + 5) begin
            x_real = $itor(i) / 1000.0;
            test_value(x_real);
        end
                
        // Summary
        total_tests = pass_count + fail_count;
        
        $display("====================================================================================");
        $display("                            TEST SUMMARY                                           ");
        $display("====================================================================================");
        $display(" Total Tests:          %4d", total_tests);
        $display(" Passed:               %4d", pass_count);
        $display(" Failed:               %4d", fail_count);
        $display(" Pass Rate:            %6.2f%%", 
                 ($itor(pass_count) / $itor(total_tests)) * 100.0);
        $display("");
        $display(" ERROR ANALYSIS (vs Exact Formula):");
        $display("   Max Absolute Error:   %10.6f", max_abs_error);
        $display("     Occurred at x:      %10.6f", worst_x);
        $display("   Max Relative Error:   %10.3f%%", max_rel_error);
        $display("     Occurred at x:      %10.6f", worst_rel_x);
        $display("");
        $display(" APPROXIMATION ERROR (HW vs Expected Formula):");
        $display("   Max HW Approx Error:  %10.6f", max_hw_error);
        $display("     Occurred at x:      %10.6f", worst_hw_x);
        $display("");
        $display(" Q10.22 Precision:     2^-22 = %e", 2.0**(-22));
        $display("                              = %f", 2.0**(-22));
        $display("====================================================================================");
        
        if (fail_count == 0) begin
            $display("                      ✓ ALL TESTS PASSED!                                       ");
        end else begin
            $display("                      ✗ SOME TESTS FAILED!                                      ");
        end
        $display("====================================================================================");
        
        $finish;
    end
    
    // Task to test a single value
    task test_value;
        input real x_val;
        real s_val, exp_exact, exp_hw, abs_err_exact, abs_err_hw, rel_err;
        string status;
        begin
            // Apply input
            x = real_to_q1022(x_val);
            #10;
            
            // Get output
            s_val = q1022_to_real(s_x);
            exp_exact = compute_expected(x_val);      // Exact formula
            exp_hw = compute_hw_approx(x_val);        // HW approximation
            
            // Error vs exact formula
            abs_err_exact = s_val - exp_exact;
            
            // Error vs HW approximation (quantization error)
            abs_err_hw = s_val - exp_hw;
            
            // Relative error vs exact
            if (exp_exact != 0.0)
                rel_err = (abs_err_exact / exp_exact) * 100.0;
            else if (abs_err_exact != 0.0)
                rel_err = 100.0;
            else
                rel_err = 0.0;
            
            // Track max errors
            if ((abs_err_exact < 0.0 ? -abs_err_exact : abs_err_exact) > max_abs_error) begin
                max_abs_error = (abs_err_exact < 0.0 ? -abs_err_exact : abs_err_exact);
                worst_x = x_val;
            end
            
            if ((rel_err < 0.0 ? -rel_err : rel_err) > max_rel_error) begin
                max_rel_error = (rel_err < 0.0 ? -rel_err : rel_err);
                worst_rel_x = x_val;
            end
            
            // Track max HW approximation error
            if ((exp_hw - exp_exact < 0.0 ? -(exp_hw - exp_exact) : exp_hw - exp_exact) > max_hw_error) begin
                max_hw_error = (exp_hw - exp_exact < 0.0 ? -(exp_hw - exp_exact) : exp_hw - exp_exact);
                worst_hw_x = x_val;
            end
            
            // Pass/fail criteria: <5% relative error OR <0.2 absolute error
            if (((rel_err < 0.0 ? -rel_err : rel_err) < 5.0) || 
                ((abs_err_exact < 0.0 ? -abs_err_exact : abs_err_exact) < 0.2)) begin
                pass_count = pass_count + 1;
                status = "PASS ";
            end else begin
                fail_count = fail_count + 1;
                status = "FAIL ";
            end
            
            $display(" %8.4f | %10.6f | %9.6f | %10.6f | %10.6f | %8.3f%% | %-10s", 
                     x_val, s_val, exp_exact, abs_err_exact, abs_err_hw, rel_err, status);
        end
    endtask

endmodule
