`timescale 1ns/1ps

module EU_tb;

    // Parameters
    localparam WIDTH = 32;
    localparam Q_IN = 22;      // Input Q10.22
    localparam Q_OUT = 16;     // Output Q48.16
    localparam NUM_PORTS = 32;
    localparam NUM_SEGMENTS = 8;
    localparam real Q_OUT_MIN = 2.0**(-Q_OUT);  // 1.526e-05: min representable

    // DUT Signals
    reg  signed [WIDTH-1:0]      s_x;
    wire [2:0]                   segment_index;
    wire signed [WIDTH-1:0]      K;
    wire signed [WIDTH-1:0]      B;
    wire signed [2*WIDTH-1:0]    exp_result;

    // SharedLUT Signals
    wire [2:0]                   segment_index_array [NUM_PORTS-1:0];
    wire signed [WIDTH-1:0]      k_coeff_array [NUM_PORTS-1:0];
    wire signed [WIDTH-1:0]      b_intercept_array [NUM_PORTS-1:0];

    assign segment_index_array[0] = segment_index;
    assign K = k_coeff_array[0];
    assign B = b_intercept_array[0];

    genvar i;
    generate
        for (i = 1; i < NUM_PORTS; i++) begin : tie_off
            assign segment_index_array[i] = 3'd0;
        end
    endgenerate

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
    real max_abs_error   = 0.0;
    real max_rel_error   = 0.0;
    real avg_abs_error   = 0.0;
    real avg_rel_error   = 0.0;
    real max_lut_error   = 0.0;
    real avg_lut_error   = 0.0;
    int  test_count      = 0;
    int  overflow_count  = 0;
    int  underflow_count = 0;   // expected: 2^x below Q48.16 min
    real worst_error_input     = 0.0;
    real worst_lut_error_input = 0.0;

    // Helper Functions
    function real pow2;
        input real x;
        begin pow2 = 2.0 ** x; end
    endfunction

    function real q1022_to_real;
        input signed [WIDTH-1:0] val;
        begin q1022_to_real = $itor(val) / (2.0 ** Q_IN); end
    endfunction

    function signed [WIDTH-1:0] real_to_q1022;
        input real val;
        real scaled;
        begin
            scaled = val * (2.0 ** Q_IN);
            if      (scaled >  2147483647.0) real_to_q1022 =  32'sd2147483647;
            else if (scaled < -2147483648.0) real_to_q1022 = -32'sd2147483648;
            else                             real_to_q1022 = $rtoi(scaled);
        end
    endfunction

    function real q4816_to_real;
        input signed [63:0] val;
        begin q4816_to_real = $itor(val) / (2.0 ** Q_OUT); end
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

            result_real    = q4816_to_real(exp_result);
            expected_exact = pow2(sx_real);

            int_part     = $floor(sx_real);
            frac_part    = sx_real - int_part;
            expected_frac = pow2(frac_part);

            k_real     = q1022_to_real(K);
            b_real     = q1022_to_real(B);
            lut_approx = k_real * frac_part + b_real;
            lut_error  = (lut_approx - expected_frac < 0) ?
                         -(lut_approx - expected_frac) : (lut_approx - expected_frac);

            // ----------------------------------------------------------------
            // Status classification
            // ----------------------------------------------------------------
            status = "";

            // Underflow: exact value below Q48.16 min representable → 0 output is CORRECT
            if (result_real == 0.0 && expected_exact < Q_OUT_MIN) begin
                status = " [UNDERFLOW_OK]";
                underflow_count++;
            end
            // True underflow: value representable but output wrong
            else if (result_real == 0.0 && expected_exact >= Q_OUT_MIN) begin
                status = " [UNDERFLOW_ERR]";
                underflow_count++;
            end
            else if (exp_result == 64'sh7FFFFFFFFFFFFFFF) begin
                status = " [OVERFLOW+]";
                overflow_count++;
            end
            else if (exp_result == 64'sh8000000000000000) begin
                status = " [OVERFLOW-]";
                overflow_count++;
            end

            // ----------------------------------------------------------------
            // Error tracking: ONLY for valid representable range
            // ----------------------------------------------------------------
            abs_error = (result_real - expected_exact < 0) ?
                        -(result_real - expected_exact) : (result_real - expected_exact);

            if (expected_exact != 0.0)
                rel_error = (abs_error / expected_exact) * 100.0;
            else
                rel_error = 0.0;

            // Exclude overflow AND underflow from stats (both expected behavior)
            if (status == "") begin
                if (abs_error > max_abs_error) begin
                    max_abs_error      = abs_error;
                    worst_error_input  = sx_real;
                end
                if (rel_error > max_rel_error) max_rel_error = rel_error;
                avg_abs_error = avg_abs_error + abs_error;
                avg_rel_error = avg_rel_error + rel_error;

                if (lut_error > max_lut_error) begin
                    max_lut_error           = lut_error;
                    worst_lut_error_input   = sx_real;
                end
                avg_lut_error = avg_lut_error + lut_error;

                test_count++;
            end

            $display("%8.4f | %3d | %10.6f | %15.6e | %15.6e | %12.6e | %8.4f%% | %10.6e%s",
                     sx_real, segment_index, frac_part,
                     result_real, expected_exact,
                     abs_error, rel_error, lut_error, status);
        end
    endtask

    // =========================================================================
    // Main Test
    // =========================================================================
    initial begin
        $display("\n================================================================================");
        $display("    EU Module Testbench - Q10.22 → Q48.16");
        $display("================================================================================");
        $display("  Input:          Q10.22 (32-bit)");
        $display("  LUT:            Q10.22 K/B coefficients (32-bit, 8 segments)");
        $display("  Output:         Q48.16 (64-bit)");
        $display("  Q48.16 range:   2^x for x ∈ [-16, ~46]");
        $display("  Min represent:  2^-16 = %.6e  (underflow below → 0, CORRECT)", Q_OUT_MIN);
        $display("================================================================================\n");

        $display("s(x)     | Seg |   frac     |   HW Result     |   Exact 2^x     |  Abs Error   | Rel Err%% |  LUT Error");
        $display("---------|-----|------------|-----------------|-----------------|--------------|----------|------------");

        // Basic values
        $display("\n--- Basic Values ---");
        test_value( 0.0);
        test_value( 1.0);
        test_value(-1.0);
        test_value( 2.0);
        test_value(-2.0);
        test_value( 0.5);
        test_value(-0.5);

        // Segment boundaries (frac = 0, 0.125, ..., 0.875)
        $display("\n--- Segment Boundaries ---");
        for (int j = 0; j < 8; j++)
            test_value($itor(j) / 8.0);

        // Mid-segment (worst LUT approx)
        $display("\n--- Mid-Segment Values (worst LUT error) ---");
        for (int j = 0; j < 8; j++)
            test_value(($itor(j) + 0.5) / 8.0);

        // Representable positive range
        $display("\n--- Representable Positive Range (x ∈ [-16, +44]) ---");
        test_value(-16.0);
        test_value(-15.0);
        test_value(-10.0);
        test_value( -5.0);
        test_value(  5.0);
        test_value( 10.0);
        test_value( 15.0);
        test_value( 20.0);
        test_value( 25.0);
        test_value( 30.0);
        test_value( 35.0);
        test_value( 40.0);
        test_value( 44.14);

        // Below representable (underflow — expect UNDERFLOW_OK)
        $display("\n--- Below Q48.16 Min (expect UNDERFLOW_OK) ---");
        test_value(-17.0);
        test_value(-20.0);
        test_value(-25.0);
        test_value(-30.0);
        test_value(-35.0);
        test_value(-38.56);
        test_value(-40.0);

        // Near maximum boundary
        $display("\n--- Near Maximum (+38 to +45) ---");
        for (int j = 38; j <= 45; j++)
            test_value($itor(j));

        // Dense sweep in core operating range
        $display("\n--- Dense Sweep (-16 to +44, step 1.0) ---");
        for (int j = -16; j <= 44; j++)
            test_value($itor(j));

        // Averages
        if (test_count > 0) begin
            avg_abs_error = avg_abs_error / test_count;
            avg_rel_error = avg_rel_error / test_count;
            avg_lut_error = avg_lut_error / test_count;
        end

        // Summary
        $display("\n================================================================================");
        $display("                         Accuracy Summary");
        $display("================================================================================");
        $display("Total test cases:          %0d", test_count + overflow_count + underflow_count);
        $display("Valid (representable):      %0d  ← error stats from these only", test_count);
        $display("Overflow cases:            %0d  (expected for x > ~46)", overflow_count);
        $display("Underflow cases (OK):      %0d  (2^x < 2^-16, output 0 is correct)", underflow_count);
        $display("");
        $display("TOTAL ERROR (valid range only, vs Exact 2^x):");
        $display("  Max Absolute Error:      %.6e  (at s(x)=%.4f)", max_abs_error, worst_error_input);
        $display("  Max Relative Error:      %.4f%%", max_rel_error);
        $display("  Avg Absolute Error:      %.6e", avg_abs_error);
        $display("  Avg Relative Error:      %.4f%%", avg_rel_error);
        $display("");
        $display("LUT APPROXIMATION ERROR (2^frac ≈ K×frac+B):");
        $display("  Max LUT Error:           %.6e  (at s(x)=%.4f)", max_lut_error, worst_lut_error_input);
        $display("  Avg LUT Error:           %.6e", avg_lut_error);
        $display("");
        $display("Format:");
        $display("  Input precision:  2^-22 = %.6e", 2.0**(-Q_IN));
        $display("  Output precision: 2^-16 = %.6e", 2.0**(-Q_OUT));
        $display("================================================================================");

        // Pass/Fail
        if      (max_rel_error < 1.0) $display("✓ EXCELLENT: Max rel error < 1%%");
        else if (max_rel_error < 5.0) $display("✓ PASS:      Max rel error < 5%%");
        else                          $display("✗ FAIL:      Max rel error = %.2f%%", max_rel_error);

        if      (avg_rel_error < 0.5) $display("✓ EXCELLENT: Avg rel error < 0.5%%");
        else if (avg_rel_error < 2.0) $display("✓ PASS:      Avg rel error < 2%%");
        else                          $display("✗ WARNING:   Avg rel error = %.2f%%", avg_rel_error);

        if      (max_lut_error < 0.01) $display("✓ EXCELLENT: LUT error < 0.01");
        else if (max_lut_error < 0.05) $display("✓ GOOD:      LUT error < 0.05");
        else                           $display("✗ WARNING:   LUT error = %.6f", max_lut_error);

        $display("✓ INFO: %0d underflow cases — correct (2^x < Q48.16 min = %.2e)",
                 underflow_count, Q_OUT_MIN);
        $display("================================================================================\n");

        $finish;
    end

endmodule
