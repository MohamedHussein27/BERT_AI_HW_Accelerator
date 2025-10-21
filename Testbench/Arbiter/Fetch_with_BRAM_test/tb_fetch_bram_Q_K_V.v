`timescale 1ns/1ps
module tb_fetch_bram_Q_K_V;

    parameter NUM_FETCHES_PER_TILE = 32;
    parameter ADDR_WIDTH = 16;
    parameter ORIGINAL_COLUMNS     = 768;   // matrix columns before transpose
    parameter ORIGINAL_ROWS        = 512;   // matrix rows before transpose
    parameter NUM_BITS             = 8;     // quantized element
    parameter DATA_WIDTH           = 256;
    parameter CLK_PERIOD           = 10;
    
    reg clk, rst_n;
    reg start_fetch, reset_addr_counter;
    reg ena, wea;
    reg [2:0] Offset_Control;
    reg [ADDR_WIDTH-1:0] addra;
    reg [DATA_WIDTH-1:0] dina;

    wire fetch_done;
    wire [ADDR_WIDTH-1:0] addrb;
    wire [DATA_WIDTH-1:0] doutb;
    

    // Clock generation
    initial begin
        clk = 0;
        forever #(CLK_PERIOD/2) clk = ~clk;  // 100 MHz clock
    end

    // DUT
    fetch_bram_Q_K_V_top #(
        .NUM_FETCHES_PER_TILE(NUM_FETCHES_PER_TILE),
        .ADDR_WIDTH(ADDR_WIDTH),
        .ORIGINAL_COLUMNS(ORIGINAL_COLUMNS),
        .ORIGINAL_ROWS(ORIGINAL_ROWS),
        .NUM_BITS(NUM_BITS),
        .DATA_WIDTH(DATA_WIDTH)
    ) u_top_1 (
        .clk(clk),
        .rst_n(rst_n),
        
        .start_fetch(start_fetch),
        .reset_addr_counter(reset_addr_counter),
        .Offset_Control(Offset_Control),
        
        .wea(wea),
        .ena(ena),
        .addra(addra),
        .dina(dina),

        .fetch_done(fetch_done),
        .doutb(doutb),
        .addrb(addrb)
    );
    
    integer i = 0;

    // Stimulus
    initial begin
        // Reset
        rst_n = 0;
        start_fetch = 0;
        reset_addr_counter = 0;
        ena = 0;
        wea = 0;
        addra = 0;
        dina = 0;
        Offset_Control = 3'b011;
        repeat(5) @(negedge clk);
        rst_n = 1;
        ena = 1;
        reset_addr_counter = 1;
        repeat(2) @(negedge clk);
        reset_addr_counter = 0;
        // =====================
        $display("Writing BRAM...");
        for (i = 0; i < 12288; i = i + 1) begin // 4096 = 512 * 8, as write ports differ from the read port 
            wea  = 1;
            dina = i * 2 + 2;     // deterministic pattern
            @(negedge clk);
            addra = addra + 1 ;   // to write in the right places to be read
        end
        wea = 0;
        repeat(5) @(negedge clk);
        
        // =====================
        $display("Starting fetch...");
        start_fetch = 1;
        @(negedge clk);
        start_fetch = 0;

        // Wait for fetch completion
        wait(fetch_done);
        $display("Fetch done at time %0t", $time);

        repeat(2) @(negedge clk);
        // fetch again
        $display("Starting fetch...");
        start_fetch = 1;
        @(negedge clk);
        start_fetch = 0;
        
        //Wait for fetch completion
        wait(fetch_done);
        $display("Fetch done at time %0t", $time);
        repeat(2) @(negedge clk);
        $stop;
    end
endmodule