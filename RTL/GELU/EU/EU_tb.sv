`timescale 1ns/1ps

module EU_tb;

    // Parameters
    localparam WIDTH = 32;
    localparam Q_IN = 22;      // Input Q10.22
    localparam Q_OUT = 16;     // Output Q48.16
    localparam NUM_PORTS = 32;
    localparam NUM_SEGMENTS = 8;

    // DUT Signals
    reg  signed [WIDTH-1:0]      s_x;           // Q10.22 input
    wire [2:0]                   segment_index;
    wire signed [WIDTH-1:0]      K;             // Q10.22 from 32-bit LUT
    wire signed [WIDTH-1:0]      B;             // Q10.22 from 32-bit LUT
    wire signed [2*WIDTH-1:0]    exp_result;    // Q48.16 output (64-bit)

    // SharedLUT Signals (use only port 0)
    wire [2:0]                   segment_index_array [NUM_PORTS-1:0];
    wire signed [WIDTH-1:0]      k_coeff_array [NUM_PORTS-1:0];
    wire signed [WIDTH-1:0]      b_intercept_array [NUM_PORTS-1:0];

    // Connect EU to LUT port 0
    assign segment_index_array[0] = segment_index;
    assign K = k_coeff_array[0];
    assign B = b_intercept_array[0];

    // Tie off unused LUT ports
    genvar i;
    generate
        for (i = 1; i < NUM_PORTS; i++) begin : tie_off
            assign segment_index_array[i] = 3'd0;
        end
    endgenerate

    // Instantiate SharedLUT (32-bit version)
    SharedLUT #(
        .W(WIDTH),
        .Q(Q_IN),
        .NUM_SEGMENTS(NUM_SEGMENTS),
        .NUM_PORTS(NUM_PORTS)
    ) lut (
        .segment_index(segment_index_array),
        .k_coeff(k_coeff_array),
        .b_intercept(b_intercept_array)
    );

    // Instantiate DUT
    EU #(
        .WIDTH(WIDTH),
        .Q_IN(Q_IN),
        .Q_OUT(Q_OUT)
    ) dut (
        .s_x(s_x),
        .segment_index(segment_index),
        .K(K),
        .B(B),
        .exp_result(exp_result)
    );

    // Accuracy Tracking
    real max_abs_error = 0.0;
    real max_rel_error = 0.0;
    real avg_abs_error = 0.0;
    real avg_rel_error = 0.0;
    real max_lut_error = 0.0;
    real avg_lut_error = 0.0;
    int  test_count = 0;
    int  overflow_count = 0;
    int  underflow_count = 0;
    real worst_error_input = 0.0;
    real worst_lut_error_input = 0.0;

    // Helper Functions
    function real pow2;
        input real x;
        begin
            pow2 = 2.0 ** x;
        end
    endfunction

    function real q1022_to_real;
        input signed [WIDTH-1:0] val;
        begin
            q1022_to_real = $itor(val) / (2.0 ** Q_IN);
        end
    endfunction

    function signed [WIDTH-1:0] real_to_q1022;
        input real val;
        real scaled;
        begin
            scaled = val * (2.0 ** Q_IN);
            if (scaled > 2147483647.0)
                real_to_q1022 = 32'sd2147483647;
            else if (scaled < -2147483648.0)
                real_to_q1022 = -32'sd2147483648;
            else
                real_to_q1022 = $rtoi(scaled);
        end
    endfunction

    // Convert Q48.16 output to real
    function real q4816_to_real;
        input signed [63:0] val;
        begin
            q4816_to_real = $itor(val) / (2.0 ** Q_OUT);
        end
    endfunction

    // Test Task
    task test_value;
        input real sx_real;
        real result_real, expected_exact, abs_error, rel_error;
        real int_part, frac_part, expected_frac, lut_approx, lut_error;
        real k_real, b_real;
        string status;
        begin
            s_x = real_to_q1022(sx_real);
            #10;
            
            result_real = q4816_to_real(exp_result);
            expected_exact = pow2(sx_real);
            
            int_part = $floor(sx_real);
            frac_part = sx_real - int_part;
            expected_frac = pow2(frac_part);
            
            k_real = q1022_to_real(K);
            b_real = q1022_to_real(B);
            lut_approx = k_real * frac_part + b_real;
            
            lut_error = lut_approx - expected_frac;
            if (lut_error < 0) lut_error = -lut_error;
            
            // Check for overflow/underflow
            status = "";
            if (result_real == 0.0 && expected_exact > 1e-4) begin
                status = " [UNDERFLOW]";
                underflow_count++;
            end else if (exp_result == 64'sh7FFFFFFFFFFFFFFF) begin
                status = " [OVERFLOW+]";
                overflow_count++;
            end else if (exp_result == 64'sh8000000000000000) begin
                status = " [OVERFLOW-]";
                overflow_count++;
            end
            
            abs_error = result_real - expected_exact;
            if (abs_error < 0) abs_error = -abs_error;
            
            if (expected_exact != 0.0) begin
                rel_error = (abs_error / expected_exact) * 100.0;
                if (rel_error < 0) rel_error = -rel_error;
            end else begin
                rel_error = 0.0;
            end
            
            if (status == "") begin
                if (abs_error > max_abs_error) begin
                    max_abs_error = abs_error;
                    worst_error_input = sx_real;
                end
                if (rel_error > max_rel_error) max_rel_error = rel_error;
                avg_abs_error = avg_abs_error + abs_error;
                avg_rel_error = avg_rel_error + rel_error;
                
                if (lut_error > max_lut_error) begin
                    max_lut_error = lut_error;
                    worst_lut_error_input = sx_real;
                end
                avg_lut_error = avg_lut_error + lut_error;
                
                test_count = test_count + 1;
            end
            
            $display("%8.4f | %3d | %10.6f | %15.6e | %15.6e | %12.6e | %8.4f%% | %10.6e%s", 
                     sx_real, segment_index, frac_part, result_real, expected_exact, 
                     abs_error, rel_error, lut_error, status);
        end
    endtask

    // Main Test
    initial begin
        $display("\n================================================================================");
        $display("    EU Module - 32-bit LUT, Q48.16 Output");
        $display("================================================================================");
        $display("Configuration:");
        $display("  Input:    Q10.22 (32-bit)");
        $display("  LUT:      Q10.22 (32-bit) K and B coefficients");
        $display("  Mantissa: 32-bit Q10.22, sign-extended to 64-bit before shift");
        $display("  Output:   Q48.16 (64-bit) - 48 integer bits, 16 fractional bits");
        $display("================================================================================\n");
        
        $display("s(x)     | Seg |   frac     |   HW Result     |   Exact 2^x     |  Total Error | Rel Err%% |  LUT Error");
        $display("---------|-----|------------|-----------------|-----------------|--------------|-----------|------------");
        
        // Basic tests
        $display("\n--- Basic Values ---");
        test_value(0.0);
        test_value(1.0);
        test_value(-1.0);
        test_value(2.0);
        test_value(-2.0);
        test_value(0.5);
        test_value(-0.5);
        
        // Segment boundaries
        $display("\n--- Segment Boundaries ---");
        for (int j = 0; j < 8; j++) begin
            test_value($itor(j) / 8.0);
        end
        
        // Mid-segment
        $display("\n--- Mid-Segment Values ---");
        for (int j = 0; j < 8; j++) begin
            test_value(($itor(j) + 0.5) / 8.0);
        end
        
        // Extended range from PolynomialUnit
        $display("\n--- Extended Range (from PolynomialUnit output) ---");
        test_value(-38.56);
        test_value(-35.0);
        test_value(-30.0);
        test_value(-25.0);
        test_value(-20.0);
        test_value(-15.0);
        test_value(-10.0);
        test_value(-5.0);
        test_value(5.0);
        test_value(10.0);
        test_value(15.0);
        test_value(20.0);
        test_value(25.0);
        test_value(30.0);
        test_value(35.0);
        test_value(40.0);
        test_value(44.14);
        
        // Fine sweep near boundaries
        $display("\n--- Near Maximum (+38 to +45) ---");
        for (int j = 38; j <= 45; j++) begin
            test_value($itor(j));
        end
        
        $display("\n--- Near Minimum (-40 to -36) ---");
        for (int j = -40; j <= -36; j++) begin
            test_value($itor(j));
        end
        
        // Dense sweep in critical range
        $display("\n--- Dense Sweep (-10 to +10, step 1.0) ---");
        for (int j = -10; j <= 10; j++) begin
            test_value($itor(j));
        end
        
        // Calculate averages
        if (test_count > 0) begin
            avg_abs_error = avg_abs_error / test_count;
            avg_rel_error = avg_rel_error / test_count;
            avg_lut_error = avg_lut_error / test_count;
        end
        
        // Summary
        $display("\n================================================================================");
        $display("                         Accuracy Summary");
        $display("================================================================================");
        $display("Total test cases:        %0d", test_count + overflow_count + underflow_count);
        $display("Valid (no overflow):     %0d", test_count);
        $display("Overflow cases:          %0d", overflow_count);
        $display("Underflow cases:         %0d", underflow_count);
        $display("");
        $display("TOTAL ERROR (vs Exact 2^x):");
        $display("  Max Absolute Error:    %.6e", max_abs_error);
        $display("    Occurred at s(x):    %.4f", worst_error_input);
        $display("  Max Relative Error:    %.4f%%", max_rel_error);
        $display("  Avg Absolute Error:    %.6e", avg_abs_error);
        $display("  Avg Relative Error:    %.4f%%", avg_rel_error);
        $display("");
        $display("LUT APPROXIMATION ERROR (2^frac ≈ K×frac + B):");
        $display("  Max LUT Error:         %.6e", max_lut_error);
        $display("    Occurred at s(x):    %.4f", worst_lut_error_input);
        $display("  Avg LUT Error:         %.6e", avg_lut_error);
        $display("================================================================================");
        $display("Format Information:");
        $display("  Input:  Q10.22 range = [-512.00, 511.99]");
        $display("  Output: Q48.16 range = [%.2e, %.2e]", -$pow(2.0, 47.0), $pow(2.0, 47.0) - $pow(2.0, -16.0));
        $display("  Q48.16 can represent 2^x for x ∈ [-16, ~46]");
        $display("  Input precision:  2^-22 = %.6e", 2.0**(-Q_IN));
        $display("  Output precision: 2^-16 = %.6e", 2.0**(-Q_OUT));
        $display("================================================================================");
        
        // Pass/Fail
        if (max_rel_error < 1.0) begin
            $display("✓ EXCELLENT: Max relative error < 1%%");
        end else if (max_rel_error < 5.0) begin
            $display("✓ PASS: Max relative error < 5%%");
        end else begin
            $display("✗ FAIL: Max relative error >= 5%% (%.2f%%)", max_rel_error);
        end
        
        if (avg_rel_error < 0.5) begin
            $display("✓ EXCELLENT: Average relative error < 0.5%%");
        end else if (avg_rel_error < 2.0) begin
            $display("✓ PASS: Average relative error < 2%%");
        end else begin
            $display("⚠ WARNING: Average relative error = %.2f%%", avg_rel_error);
        end
        
        if (max_lut_error < 0.01) begin
            $display("✓ EXCELLENT: LUT approximation error < 0.01");
        end else if (max_lut_error < 0.05) begin
            $display("✓ GOOD: LUT approximation error < 0.05");
        end else begin
            $display("⚠ WARNING: LUT approximation error = %.6f", max_lut_error);
        end
        
        if (overflow_count == 0) begin
            $display("✓ PASS: No overflow in tested range");
        end else begin
            $display("⚠ WARNING: %0d overflow cases", overflow_count);
        end
        
        if (underflow_count == 0) begin
            $display("✓ PASS: No underflow in tested range");
        end else begin
            $display("⚠ INFO: %0d underflow cases (expected for very negative values)", underflow_count);
        end
        
        $display("================================================================================\n");
        
        $finish;
    end

endmodule
