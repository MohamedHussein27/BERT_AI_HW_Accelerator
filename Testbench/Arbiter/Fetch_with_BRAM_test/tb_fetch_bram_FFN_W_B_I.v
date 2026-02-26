`timescale 1ns/1ps
module tb_fetch_bram_FFN_W_B_I;

    parameter ADDR_WIDTH       = 16;
    parameter ORIGINAL_COLUMNS = 768;
    parameter ORIGINAL_ROWS    = 512;
    parameter NUM_BITS         = 8;
    parameter DATA_WIDTH       = 256;
    parameter CLK_PERIOD       = 10;

    reg clk, rst_n;
    reg start_fetch, reset_addr_counter, Double_buffering;
    reg [3:0] Buffer_Select;
    reg Tiles_Control;
    reg ena, wea;
    reg [13:0] addra;
    reg [DATA_WIDTH-1:0] dina;

    wire fetch_done;
    wire busy;
    wire [DATA_WIDTH-1:0] doutb;
    wire [ADDR_WIDTH-1:0] addrb;

    // =====================
    // Clock generation
    // =====================
    initial begin
        clk = 0;
        forever #(CLK_PERIOD/2) clk = ~clk;
    end

    // =====================
    // DUT
    // =====================
    fetch_bram_FFN_W_B_I_top #(
        .ADDR_WIDTH(ADDR_WIDTH),
        .ORIGINAL_COLUMNS(ORIGINAL_COLUMNS),
        .ORIGINAL_ROWS(ORIGINAL_ROWS),
        .NUM_BITS(NUM_BITS),
        .DATA_WIDTH(DATA_WIDTH)
    ) u_top (
        .clk(clk),
        .rst_n(rst_n),

        .start_fetch(start_fetch),
        .reset_addr_counter(reset_addr_counter),
        .Buffer_Select(Buffer_Select),
        .Tiles_Control(Tiles_Control),
        .Double_buffering(Double_buffering),

        .wea(wea),
        .ena(ena),
        .addra(addra),
        .dina(dina),

        .fetch_done(fetch_done),
        .doutb(doutb),
        .addrb(addrb),
        .busy(busy)
    );

    integer i;

    // =====================
    // Stimulus
    // =====================
    initial begin

        // Reset
        rst_n = 0;
        start_fetch = 0;
        reset_addr_counter = 0;
        ena = 0;
        wea = 0;
        addra = 0;
        dina = 0;
        Buffer_Select = 4'b0000;
        Tiles_Control = 1'b1;
        Double_buffering = 0;

        repeat(5) @(negedge clk);
        rst_n = 1;
        ena = 1;

        // Reset internal pointer
        reset_addr_counter = 1;
        repeat(2) @(negedge clk);
        reset_addr_counter = 0;

        // =====================
        // Write BRAM
        // =====================
        $display("Writing FFN_W_B_I_buffer...");
        for (i = 0; i < 10000; i = i + 1) begin
            wea  = 1;
            dina = i + 32'h1000;
            @(negedge clk);
            addra = addra + 1;
        end
        wea = 0;

        repeat(5) @(negedge clk);

        // =====================
        // Fetch 1
        // =====================
        $display("Starting FFN fetch...");
        start_fetch = 1;
        @(negedge clk);
        start_fetch = 0;

        wait(fetch_done);
        $display("Fetch done at time %0t", $time);

        // =====================
        // Change tiling mode
        // =====================
        Tiles_Control = 1'b0;
        reset_addr_counter = 1;
        repeat(2) @(negedge clk);
        reset_addr_counter = 0;

        $display("Starting second FFN fetch...");
        start_fetch = 1;
        @(negedge clk);
        start_fetch = 0;

        wait(fetch_done);
        $display("Second fetch done at time %0t", $time);

        // =====================
        // Double buffering test
        // =====================
        Double_buffering = 1'b1;

        reset_addr_counter = 1;
        repeat(2) @(negedge clk);
        reset_addr_counter = 0;

        $display("Starting double-buffer FFN fetch...");
        start_fetch = 1;
        @(negedge clk);
        start_fetch = 0;

        wait(fetch_done);
        $display("Double-buffer FFN fetch done at time %0t", $time);

        repeat(2) @(negedge clk);
        $stop;
    end

endmodule
