`timescale 1ns/1ps

module tb_stream_skewer();

    parameter DATAWIDTH = 8;
    parameter N_SIZE = 4;
    parameter CLK_PERIOD = 10;
    

    logic clk;
    logic rst_n;
    logic valid_in;
    logic [DATAWIDTH-1:0] in_A [N_SIZE];
    logic [DATAWIDTH-1:0] out [N_SIZE];

    skew_buffer #(
        .N_SIZE(N_SIZE),
        .DATAWIDTH(DATAWIDTH)
    ) uut (
        .clk(clk),
        .rst_n(rst_n),
        .valid_in(valid_in),
        .in_A(in_A),
        .out(out)
    );

    // Clock generation
    initial begin
        clk = 0;
        forever #(CLK_PERIOD/2) clk = ~clk;
    end

    initial begin
        // Init
        rst_n = 0; valid_in = 0;
        for (int i=0; i<N_SIZE; i++) in_A[i] = 0;

        #20 rst_n = 1;
        $display("\n--- Starting Skew Test with Valid Logic (N=4) ---");
        $display("Time | Val | In0 In1 In2 In3 | Out0 Out1 Out2 Out3");
        $display("--------------------------------------------------");

        //feed valid data
        @(negedge clk); valid_in = 0; in_A = '{99, 99, 99, 99};
        @(negedge clk); valid_in = 1; in_A = '{1, 2, 3, 4};
        @(negedge clk); valid_in = 1; in_A = '{5, 6, 7, 8};
        @(negedge clk); valid_in = 0; in_A = '{99, 99, 99, 99};
        @(negedge clk); valid_in = 1; in_A = '{9, 10, 11, 12};
        @(negedge clk); valid_in = 0; in_A = '{99, 99, 99, 99};
        @(negedge clk); valid_in = 1; in_A = '{13, 14, 15, 16};
        @(negedge clk); valid_in = 0; in_A = '{99, 99, 99, 99};

        repeat(5) begin
            @(negedge clk); valid_in = 1; in_A = '{0, 0, 0, 0};
        end;

        $display("--------------------------------------------------");
        $finish;
    end

    always @(negedge clk) begin
        if (rst_n) begin
            $display("%4t |  %1b  | %3d %3d %3d %3d | %4d %4d %4d %4d", 
                     $time, valid_in,
                     in_A[0], in_A[1], in_A[2], in_A[3],
                     out[0], out[1], out[2], out[3]);
        end
    end
endmodule