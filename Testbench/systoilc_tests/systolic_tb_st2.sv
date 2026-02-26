`timescale 1ns / 1ps
module systolic_tb_st2;

    parameter CLK_PERIOD = 10;
    parameter DATAWIDTH = 8;
    parameter DATAWIDTH_output = 32;
    parameter N_SIZE = 32;
    parameter num_of_raws = 512;
    parameter BUS_WIDTH = 256;
    parameter ADDR_WIDTH  = 10;
    parameter DEPTH = 543;
    
    // DUT signals
    logic [BUS_WIDTH-1:0] in_A;
    logic [BUS_WIDTH-1:0] weights;
    logic clk;
    logic rst_n;
    logic valid_in;
    logic load_weight;
    logic last_tile;
    logic first_iteration; 
    logic [ADDR_WIDTH-1:0] rd_addr_outbuffer;
    // Status signals
    logic ready;
    logic busy;
    logic done;
    logic [(DATAWIDTH_output*N_SIZE)-1:0] out_data_outbuffer;
    logic signed [DATAWIDTH-1:0] pp;
    integer file;
    
    // DUT
    systolic_top #(
        .DATAWIDTH(DATAWIDTH),
        .DATAWIDTH_output(DATAWIDTH_output),
        .N_SIZE(N_SIZE),
        .num_of_raws(num_of_raws),
        .BUS_WIDTH(BUS_WIDTH),
        .ADDR_WIDTH(ADDR_WIDTH),
        .DEPTH(DEPTH)
    ) dut (
        .in_A(in_A),
        .weights(weights),
        .clk(clk),
        .rst_n(rst_n),
        .valid_in(valid_in),
        .load_weight(load_weight),
        .last_tile(last_tile),
        .first_iteration(first_iteration),
        .rd_addr_outbuffer(rd_addr_outbuffer),
        .ready(ready),
        .busy(busy),
        .done(done),
        .out_data_outbuffer(out_data_outbuffer)
    );
    
    // Clock
    initial begin
        clk = 0;
        forever #(CLK_PERIOD/2) clk = ~clk;
    end
    
    // Main test - VERY SIMPLE
    initial begin   
        // 1. OPEN THE FILE FIRST
        file = $fopen("output_buffer.txt", "w");
        // 2. CHECK IF OPENED
        if (file == 0) begin
            $display("ERROR: File could not be opened!");
            $finish;
        end
        // Initialize
        in_A = 0;
        weights = 0;
        valid_in = 0;
        load_weight = 0;
        last_tile = 0;
        first_iteration = 0;
        rd_addr_outbuffer = 0;
        rst_n = 0;
        pp = 0;
        @(negedge clk);

        $display("[%0t] Loading ALL ONES weights...", $time);
        rst_n = 1;
        load_weight = 1;
        
        for (int row = 0; row < 32; row++) begin
            weights = {32{8'sd1}};  // All 32 weights = 1 (signed)
            @(negedge clk);
        end
        
        load_weight = 0;
        @(negedge clk);
        $display("[%0t] Weights loaded\n", $time);
        first_iteration = 1;
        last_tile = 0;
        // Small delay
        repeat(2) @(negedge clk);
        
        $display("[%0t] Sending ALL ONES input (512 rows)...", $time);
        valid_in = 1;
    
        for (int row = 0; row < num_of_raws; row++) begin
            pp = pp+1;
            if (pp == 253) pp = 0;
            in_A = {32{pp}};  // All 32 inputs = pp (signed byte)
            @(negedge clk);
        end
        $display("[%0t] Input sent\n", $time);
        
        // Drain
        $display("[%0t] Waiting for done signal...", $time);
         while (!done) begin
             in_A = '0;
             @(negedge clk);
         end
        valid_in = 0;
        $display("[%0t] Done signal received!\n", $time);    
 
        // Wait extra
        repeat(2) @(negedge clk);
        
        first_iteration = 0;
        last_tile = 0;  
        valid_in = 1; 

        pp = 0;
        for (int row = 0; row < num_of_raws; row++) begin
            pp = pp+1;
            if (pp == 253) pp = 0;
            in_A = {32{pp}};  // All 32 inputs = pp (signed byte)
            @(negedge clk);
        end
        
        $display("[%0t] Input sent\n", $time);
        
        // Drain
        $display("[%0t] Waiting for done signal...", $time);
         while (!done) begin
             in_A = '0;
             @(negedge clk);
         end
        valid_in = 0;
        $display("[%0t] Done signal received!\n", $time);

        repeat(2) @(negedge clk);
        
        first_iteration = 0;
        last_tile = 1;  
        valid_in = 1; 

        pp = 0;
        for (int row = 0; row < num_of_raws; row++) begin
            pp = pp+1;
            if (pp == 253) pp = 0;
            in_A = {32{pp}};  // All 32 inputs = pp (signed byte)
            @(negedge clk);
        end
        
        $display("[%0t] Input sent\n", $time);
        
        // Drain
        $display("[%0t] Waiting for done signal...", $time);
         while (!done) begin
             in_A = '0;
             @(negedge clk);
         end
        valid_in = 0;
        $display("[%0t] Done signal received!\n", $time);

        // Read results
        $display("[%0t] Reading results...", $time);
        
        for (int row = 0; row < DEPTH; row++) begin    
            rd_addr_outbuffer = row;
             @(negedge clk);
            $fwrite(file, "Row %0d: ", row);
            
            // WRITE ALL 32 VALUES
            for (int col = 0; col < N_SIZE; col++) begin

                logic signed [DATAWIDTH_output-1:0] value;
                value = out_data_outbuffer[col*DATAWIDTH_output +: DATAWIDTH_output];
                $write("%0d ", value);
                $fwrite(file, "%0d ", value);   
            end
            
            $write("\n");
            $fwrite(file, "\n");
        end
        // CLOSE FILE
        $fclose(file);
        $display("Results written to output_buffer.txt");
        $stop;
    end
endmodule