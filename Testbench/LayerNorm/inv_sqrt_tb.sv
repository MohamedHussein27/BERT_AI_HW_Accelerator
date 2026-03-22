`timescale 1ns/1ps

module inv_sqrt_tb;

    parameter DATAWIDTH = 32;
    parameter CLK_PERIOD = 10;

    // DUT Signals
    logic clk;
    logic rst_n;
    logic valid_in;
    logic signed [DATAWIDTH-1:0] data_in;
    logic signed [DATAWIDTH-1:0] data_out;
    logic [31:0] rand_val;
    logic valid_out;
    logic error;
    logic busy;

    // Statistics
    real max_error = 0.0;
    real current_error = 0.0;
    int  pass_count = 0;

    // Helper: Q5.26 to Real
    function real q5_26_to_real(input logic signed [31:0] val);
        return real'(val) / (2.0**26);
    endfunction

    inv_sqrt #(.DATAWIDTH(DATAWIDTH)) dut (.*); // Using .* shortcut for same-named signals

    initial begin
        clk = 0;
        forever #(CLK_PERIOD/2) clk = ~clk;
    end

    initial begin
        rst_n = 0;
        valid_in = 0;
        #(CLK_PERIOD * 5);
        rst_n = 1;
        
        $display("\n=====================================================");
        $display("   STARTING RANDOMIZED STRESS TEST (100 ITERATIONS)  ");
        $display("=====================================================\n");

        for (int i = 0; i < 100; i++) begin
            // Generate random positive value in Q5.26 range
            // We focus on values between 0.01 and 16.0 (common in BERT Variance)
            rand_val = $urandom_range(32'h00068DB8, 32'h40000000); 
            
            run_test(rand_val);
        end

        $display("\n=====================================================");
        $display("   FINAL REPORT");
        $display("   Tests Run: 100");
        $display("   Maximum Error Found: %f%%", max_error * 100.0);
        
        if (max_error < 0.01) 
            $display("   RESULT: PASSED (High Precision)");
        else 
            $display("   RESULT: CHECK LUT (Precision low)");
        $display("=====================================================\n");
        $finish;
    end

    task run_test(input logic signed [31:0] val);
        real real_in, real_out, expected;
        begin
            wait(!busy);
            @(posedge clk);
            data_in = val;
            valid_in = 1;
            @(posedge clk);
            valid_in = 0;

            wait(valid_out);
            
            real_in  = q5_26_to_real(val);
            real_out = q5_26_to_real(data_out);
            expected = 1.0 / $sqrt(real_in);

            // Calculate Relative Error: |expected - actual| / expected
            current_error = (expected > real_out) ? (expected - real_out) : (real_out - expected);
            current_error = current_error / expected;

            if (current_error > max_error) max_error = current_error;

            $display("[%0d] In: %f | Out: %f | Err: %f%%", pass_count, real_in, real_out, current_error * 100.0);
            pass_count++;
            
            @(posedge clk);
        end
    endtask

endmodule