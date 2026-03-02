`timescale 1ns/1ps
module tb_fetch_bram_SV_H;

    parameter ADDR_WIDTH           = 15 ;
    parameter ORIGINAL_COLUMNS     = 768;
    parameter ORIGINAL_ROWS        = 512;
    parameter NUM_BITS             = 8  ;
    parameter DATA_WIDTH           = 256;
    parameter CLK_PERIOD           = 10 ;

    reg clk, rst_n;
    reg start_fetch, reset_addr_counter, Double_buffering;
    reg [3:0] Buffer_Select;
    reg Tiles_Control;
    reg ena, wea;
    reg [ADDR_WIDTH-1:0] addra;
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
    // DUT (UPDATED TOP MODULE)
    // =====================
    fetch_bram_SV_H_top #(
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

    integer i = 0;

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
        Buffer_Select = 4'b0111;
        Tiles_Control = 1'b0;               // KT*Q treated as input while v treated as Weight
        Double_buffering = 0;

        repeat(5) @(negedge clk);
        rst_n = 1;
        ena = 1;

        reset_addr_counter = 1;
        repeat(2) @(negedge clk);
        reset_addr_counter = 0;

        // =====================
        $display("Writing SV_H_intermediate_buffer...");
        for (i = 0; i < 24576; i = i + 1) begin
            wea  = 1;
            dina = i * 2 + 2;
            @(negedge clk);
            addra = addra + 1;
        end
        wea = 0;

        repeat(5) @(negedge clk);

        // =====================
        // Fetch 5 times
        // =====================
        for (i = 0; i <5; i = i + 1) begin
            $display("Starting fetch...");
            start_fetch = 1;
            @(negedge clk);
            start_fetch = 0;
    
            wait(fetch_done);
            repeat(2) @(negedge clk);
            $display("Fetch done at time %0t", $time);
        end

        repeat(2) @(negedge clk);
        $stop;
    end

endmodule