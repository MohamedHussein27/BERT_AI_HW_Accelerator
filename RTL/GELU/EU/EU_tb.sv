`timescale 1ns/1ps

module EU_tb;

    // Parameters
    localparam int Q = 26;
    localparam int W = 32;

    // DUT I/O
    logic clk;
    logic rst_n;
    logic valid_in;
    logic signed [W-1:0] x;
    logic signed [W-1:0] EU_out;

    // Instantiate DUT
    EU #(
        .Q(Q),
        .W(W)
    ) dut (
        .clk(clk),
        .rst_n(rst_n),
        .valid_in(valid_in),
        .x(x),
        .EU_out(EU_out)
    );

    // Clock generation (10 ns period)
    initial clk = 0;
    always #5 clk = ~clk;

    // Test variables
    real x_real;
    real EU_real;

    // Test sequence
    initial begin
        // Initialize
        rst_n = 0;
        valid_in = 0;
        x = '0;

        // Reset pulse
        #20;
        rst_n = 1;

        // Apply test: x = 2.5 (Q5.26)
        x_real = 2.5;
        x = $rtoi(x_real * (1 << Q));   // convert float → -point

        #10 valid_in = 1;
        #10 valid_in = 0; // only 1 cycle of valid

        // Wait a few cycles for output
        #100;

        // Convert back to real
        EU_real = $itor(EU_out) / real'(1 << Q);

        // Display resultsfixed
        $display("----------------------------------------------------");
        $display("Input  (x): real = %f, Q5.26 = 0x%08h", x_real, x);
        $display("Output (EU_out): Q5.26 = 0x%08h, real ≈ %f", EU_out, EU_real);
        $display("----------------------------------------------------");

        $stop;
    end
endmodule
