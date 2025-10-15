`timescale 1ns/1ps
module tb_write_bram;

    // System signals
    reg clk, rst_n;
    reg start_write, reset_addr_counter;

    // Systolic Array output simulation
    reg [255:0] sa_out_data;

    // Read interface (Port B)
    reg read_en;
    reg [15:0] read_addr;
    wire [255:0] doutb;

    // Status
    wire write_done;
    wire [15:0] current_addr;

    // Clock generation
    initial begin
        clk = 0;
        forever #5 clk = ~clk; // 100 MHz
    end

    // DUT
    write_bram_top u_top (
        .clk(clk),
        .rst_n(rst_n),

        .start_write(start_write),
        .reset_addr_counter(reset_addr_counter),

        .sa_out_data(sa_out_data),

        .read_en(read_en),
        .read_addr(read_addr),
        .doutb(doutb),

        .write_done(write_done),
        .current_addr(current_addr)
    );

    // Local mirror to store expected BRAM contents
    reg [255:0] expected_data [0:255];
    integer i;

    // ======================================
    // Main stimulus process
    // ======================================
    initial begin
        // Initial values
        rst_n = 0;
        start_write = 0;
        reset_addr_counter = 0;
        sa_out_data = 0;
        read_en = 0;
        read_addr = 0;
        @(negedge clk);
        rst_n = 1;
        start_write = 1;
        // =====================
        // 1ï¸?âƒ£ Simulate SA writing 16 tiles (NUM_WRITES_PER_TILE)
        // =====================
        $display("\n--- Writing data to BRAM via write logic ---");
        
        for (i = 0; i < 36864; i = i + 1) begin
            // Create distinct 256-bit data pattern for each write
            sa_out_data = i + 2; // repeating byte pattern for visibility

            // Save expected data at the corresponding BRAM address
            expected_data[i] = sa_out_data;
            
            @(negedge clk);
            if(i==0) start_write = 0;
        end
        
        //wait(write_done);
        $display("âœ… Write done for tile %0d, address=%0d, data=%h", i, current_addr, sa_out_data);
        // Start writing this tile
        
        

        #100;

        // =====================
        // 2ï¸?âƒ£ Read back & verify written data
        // =====================
        $display("\n--- Reading back from BRAM and verifying ---");
        read_en = 1;

        for (i = 0; i < 384; i = i + 1) begin
            // Each write increments by stride=23
            read_addr = i * 24;

            @(posedge clk);
            #2; // allow BRAM output to settle

            if (doutb === expected_data[i])
                $display("âœ… Readback OK at addr %0d : %h", read_addr, doutb);
            else begin
                $display("â?Œ MISMATCH at addr %0d", read_addr);
                $display("   Expected: %h", expected_data[i]);
                $display("   Got     : %h", doutb);
            end
        end

        #100;
        $display("\nSimulation complete.");
        $stop;
    end

endmodule
