
`timescale 1ns/1ps
module tb_fetch_bram;

    reg clk, rst_n;
    reg start_fetch, reset_addr_counter;
    reg ena, wea;
    reg [13:0] addra;
    reg [31:0] dina;

    wire fetch_done;
    wire [255:0] doutb;
    wire [10:0] addrb;

    // Clock
    initial begin
        clk = 0;
        forever #5 clk = ~clk;
    end

    // DUT
    fetch_bram_tb_top u_top (
        .clk(clk),
        .rst_n(rst_n),
        .start_fetch(start_fetch),
        .reset_addr_counter(reset_addr_counter),

        .wea(wea),
        .ena(ena),
        .addra(addra),
        .dina(dina),

        .fetch_done(fetch_done),
        .doutb(doutb),
        .addrb(addrb)
    );

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
        #50;
        rst_n = 1;
        ena = 1;

        // =====================
        // 1️⃣ Write some values to BRAM
        // =====================
        $display("Writing BRAM...");
        repeat (4) begin
            wea = 1;
            dina = $random;
            addra = addra + 1;
            #10;
        end
        wea = 0;
        #50;

        // =====================
        // 2️⃣ Start fetch logic to read
        // =====================
        $display("Starting fetch...");
        start_fetch = 1;
        #10 start_fetch = 0;

        // Wait until done
        wait(fetch_done);
        $display("Fetch done at time %0t", $time);

        #1000;
        $finish;
    end

endmodule