`timescale 1ns/1ps

module EU_tb;

    // Parameters
    localparam int Q = 26;
    localparam int W = 32;
    localparam int INT_WIDTH = 5;

    // DUT I/O
    logic                        clk;
    logic                        rst_n;
    logic                        valid_in;
    logic [INT_WIDTH-1:0]        integer_part;
    logic [Q-1:0]                frac_part;
    logic signed [W-1:0]         k_coeff;
    logic signed [W-1:0]         b_intercept;
    logic                        valid_out;
    logic signed [W-1:0]         exp_result;

    // LUT values matching segment-based interpolation
    localparam logic signed [W-1:0] k_fixed [0:7] = '{
        32'h02E57078, // 0.724062
        32'h03288B9B, // 0.789595
        32'h0371B996, // 0.861060
        32'h03C18722, // 0.938992
        32'h04188DB7, // 1.023978
        32'h047774AE, // 1.116656
        32'h04DEF287, // 1.217722
        32'h054FCE46 // 1.327935
    };

    localparam logic signed [W-1:0] b_fixed [0:7] = '{
        32'h04000000, // 1.000000
        32'h03F79C9B, // 0.991808
        32'h03E5511D, // 0.973942
        32'h03C76408, // 0.944718
        32'h039BE0BD, // 0.902224
        32'h03609063, // 0.844301
        32'h0312F200, // 0.768501
        32'h02B031B9 // 0.672065
    };

    // Instantiate DUT
    EU #(
        .Q(Q),
        .W(W),
        .INT_WIDTH(INT_WIDTH)
    ) dut (
        .clk(clk),
        .rst_n(rst_n),
        .valid_in(valid_in),
        .integer_part(integer_part),
        .frac_part(frac_part),
        .k_coeff(k_coeff),
        .b_intercept(b_intercept),
        .valid_out(valid_out),
        .exp_result(exp_result)
    );

    // Clock generation (10 ns period)
    initial clk = 0;
    always #5 clk = ~clk;

    // Test variables
    real x_real;
    real exp_real;
    real expected_real;
    real error_percent;
    logic signed [W-1:0] x_fixed;
    logic [2:0] segment;
    int test_num;

    // Task to run a single test
    task automatic run_test(real x_input);
        begin
            test_num++;
            x_real = x_input;
            x_fixed = $rtoi(x_real * (1 << Q));
            
            // Extract integer and fractional parts
            integer_part = x_fixed[W-1:Q];
            frac_part = x_fixed[Q-1:0];
            
            // Determine segment from top 3 fractional bits
            segment = frac_part[Q-1:Q-3];
            
            // Select appropriate LUT values
            k_coeff = k_fixed[segment];
            b_intercept = b_fixed[segment];

            #10 valid_in = 1;
            #10 valid_in = 0;

            // Wait for output to be valid
            wait(valid_out);
            #10;

            // Convert back to real
            exp_real = $itor(exp_result) / real'(1 << Q);
            expected_real = 2.0 ** x_real;
            
            // Calculate error percentage
            if (expected_real != 0.0)
                error_percent = ((exp_real - expected_real) / expected_real) * 100.0;
            else
                error_percent = 0.0;

            // Display results in tabular format
            $display("| %2d | %7.3f | 0x%08h | %12.6f | %12.6f | %8.4f%% |",
                     test_num, x_real, exp_result, exp_real, expected_real, error_percent);
            
            #20; // Gap between tests
        end
    endtask

    // Test sequence
    initial begin
        // Initialize
        rst_n = 0;
        valid_in = 0;
        integer_part = '0;
        frac_part = '0;
        k_coeff = '0;
        b_intercept = '0;
        test_num = 0;

        // Reset pulse
        #20;
        rst_n = 1;
        #10;

        // Display header
        $display("\n========================================================================");
        $display("                    EU Module Test Results");
        $display("========================================================================");
        $display("| Test |  Input  |   Output Hex   |   Output    |   Expected  |  Error   |");
        $display("|  No  |   (x)   |   (Q5.26)      |   (Real)    |   2^x       |    (%%)   |");
        $display("|------|---------|----------------|-------------|-------------|----------|");

        // Test cases covering different ranges
        
        // Small positive values
        run_test(0.125);
        run_test(0.25);
        run_test(0.5);
        run_test(0.75);
        
        // Around 1.0
        run_test(1.0);
        run_test(1.25);
        run_test(1.5);
        run_test(1.75);
        
        // Range 2.0 to 3.0
        run_test(2.0);
        run_test(2.125);
        run_test(2.25);
        run_test(2.375);
        run_test(2.5);
        run_test(2.625);
        run_test(2.75);
        run_test(2.875);
        run_test(3.0);
        
        // Range 3.0 to 4.0
        run_test(3.125);
        run_test(3.25);
        run_test(3.5);
        run_test(3.75);
        run_test(4.0);
        run_test(4.125);
        run_test(4.25);
        run_test(4.5);
        run_test(4.75);
        
        // Upper range 4.5 to 5.0
        run_test(4.875);
        run_test(4.9375);
        
        // Small negative values
        run_test(-0.125);
        run_test(-0.25);
        run_test(-0.5);
        run_test(-0.75);
        
        // Around -1.0
        run_test(-1.0);
        run_test(-1.25);
        run_test(-1.5);
        run_test(-1.75);
        
        // Range -2.0 to -3.0
        run_test(-2.0);
        run_test(-2.125);
        run_test(-2.25);
        run_test(-2.375);
        run_test(-2.5);
        run_test(-2.625);
        run_test(-2.75);
        run_test(-2.875);
        run_test(-3.0);
        
        // Range -3.0 to -4.0
        run_test(-3.125);
        run_test(-3.25);
        run_test(-3.5);
        run_test(-3.75);
        run_test(-4.0);
        run_test(-4.125);
        run_test(-4.25);
        run_test(-4.5);
        run_test(-4.75);
        
        // Lower range -4.5 to -5.0
        run_test(-4.875);
        run_test(-4.9375);
        
        // Edge cases
        run_test(0.0);
        
        // Fractional edge cases (segment boundaries)
        run_test(0.875);  // Near segment boundary
        run_test(1.125);
        run_test(3.375);
        run_test(4.625);
        run_test(-0.875);
        run_test(-3.375);
        run_test(-4.625);
        
        $display("========================================================================\n");
        
        $display("\nAll tests completed!");
        $stop;
    end
endmodule
