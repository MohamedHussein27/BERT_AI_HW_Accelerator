`timescale 1ns/1ps
module tb_fetch_bram;
    
    reg clk, rst_n;
    reg start_fetch, reset_addr_counter;
    //reg [1:0] buffer_select;
    reg ena, wea;
    reg [13:0] addra;
    reg [31:0] dina;

    wire fetch_done;
    wire [255:0] doutb;
    wire [10:0] addrb;

    // Clock generation
    initial begin
        clk = 0;
        forever #5 clk = ~clk;  // 100 MHz clock
    end

    // DUT
    fetch_bram_top #(
        .NUM_FETCHES_PER_TILE(512),
        .ADDR_WIDTH(11),
        .FETCH_START_OFFSET(112) ) u_top (
        .clk(clk),
        .rst_n(rst_n),
        .start_fetch(start_fetch),
        .reset_addr_counter(reset_addr_counter),
        //.buffer_select(buffer_select),

        .wea(wea),
        .ena(ena),
        .addra(addra),
        .dina(dina),

        .fetch_done(fetch_done),
        .doutb(doutb),
        .addrb(addrb)
    );

    // Local memory mirror to check correctness
    reg [31:0] written_values [0:255];  // same size as number of writes
    integer i;

    // Stimulus
    initial begin
        // Reset
        rst_n = 0;
        start_fetch = 0;
        reset_addr_counter = 0;
        ena = 0;
        //buffer_select = 2'b00;
        wea = 0;
        addra = 896;  // 112 * 8 (difference bet. the write depth and read depth) to test the I buffer
        dina = 0;
        #50;
        rst_n = 1;
        ena = 1;
        reset_addr_counter = 1;
        @(negedge clk);
        @(negedge clk);
        reset_addr_counter = 0;
        //buffer_select = 2'b00;
        // =====================
        // 1ï¸�âƒ£ Write values to BRAM
        // =====================
        $display("Writing BRAM...");
        for (i = 0; i < 4096; i = i + 1) begin // 4096 = 512 * 8, as write ports differ from the read port 
            wea  = 1;
            dina = i * 2 + 2;     // deterministic pattern
            written_values[i] = dina;
            #10;
            addra = addra + 1 ;   // to write in the right places to be read
        end
        wea = 0;
        #50;

        // =====================
        // 2ï¸�âƒ£ Start fetch logic (read)
        // =====================
        $display("Starting fetch...");
        start_fetch = 1;
        #10;
        start_fetch = 0;

        // Wait for fetch completion
        wait(fetch_done);
        $display("Fetch done at time %0t", $time);

        @(negedge clk);
        @(negedge clk);
        // fetch again
        $display("Starting fetch...");
        start_fetch = 1;
        #10;
        start_fetch = 0;

        // Wait for fetch completion
        wait(fetch_done);
        $display("Fetch done at time %0t", $time);
        @(negedge clk);
        @(negedge clk);
        
        // fetch again
        /*$display("Starting fetch...");
        start_fetch = 1;
        #10;
        start_fetch = 0;

        // Wait for fetch completion
        wait(fetch_done);
        $display("Fetch done at time %0t", $time);
        @(negedge clk);
        @(negedge clk);*/
        

        
         //=====================
         //3ï¸�âƒ£ Verification
        //=====================
        //Each doutb should equal the concatenation of 8 consecutive 32-bit writes:
        //doutb = {w[7], w[6], w[5], w[4], w[3], w[2], w[1], w[0]}
        //check_bram_output();
        $stop;
    end

    // =====================
    // Verification task
    // =====================
    task check_bram_output;
        reg [255:0] expected;
        integer group, j;
        begin
            for (group = 0; group < 32; group = group + 1) begin
                expected = {written_values[group*8 + 7],
                            written_values[group*8 + 6],
                            written_values[group*8 + 5],
                            written_values[group*8 + 4],
                            written_values[group*8 + 3],
                            written_values[group*8 + 2],
                            written_values[group*8 + 1],
                            written_values[group*8 + 0]};

                // Wait for BRAM address to match this group (simulate sequential read)
                wait(addrb == group);
                @(posedge clk);

                if (doutb === expected)
                    $display("Correct BRAM output for group %0d (addr=%0d)", group, addrb);
                else begin
                    $display("MISMATCH at group %0d (addr=%0d)", group, addrb);
                    $display("   Expected: %h", expected);
                    $display("   Got     : %h", doutb);
                end
            end
        end
    endtask

endmodule