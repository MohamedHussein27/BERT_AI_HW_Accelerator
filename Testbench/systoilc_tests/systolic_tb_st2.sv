`timescale 1ns / 1ps
module systolic_tb_st2;

    parameter CLK_PERIOD      = 10;
    parameter DATAWIDTH        = 8;
    parameter DATAWIDTH_output = 32;
    parameter N_SIZE           = 32;
    parameter num_of_raws      = 512;
    parameter BUS_WIDTH        = 256;
    parameter ADDR_WIDTH       = 10;
    parameter DEPTH            = 512;

    // ------------------------------------------------------------------ DUT IO
    logic [BUS_WIDTH-1:0] in_A;
    logic [BUS_WIDTH-1:0] weights;
    logic clk;
    logic rst_n;
    logic valid_in;
    logic load_weight;
    logic last_tile;
    logic first_iteration;

    // Status outputs
    logic ready;
    logic busy;
    logic done;
    logic valid_out;                                   // replaces rd_addr / out_data_outbuffer

    logic signed [(DATAWIDTH_output*N_SIZE)-1:0] data_out;   // live output bus

    // ---------------------------------------------------------------- helpers
    logic signed [DATAWIDTH-1:0] pp;
    integer file;
    int     row_capture;                               // counts rows written to file

    // -------------------------------------------------------------------- DUT
    systolic_top #(
        .DATAWIDTH       (DATAWIDTH),
        .DATAWIDTH_output(DATAWIDTH_output),
        .N_SIZE          (N_SIZE),
        .num_of_raws     (num_of_raws),
        .BUS_WIDTH       (BUS_WIDTH),
        .ADDR_WIDTH      (ADDR_WIDTH)
    ) dut (
        .in_A            (in_A),
        .weights         (weights),
        .clk             (clk),
        .rst_n           (rst_n),
        .valid_in        (valid_in),
        .load_weight     (load_weight),
        .last_tile       (last_tile),
        .first_iteration (first_iteration),
        .ready           (ready),
        .busy            (busy),
        .done            (done),
        .valid_out       (valid_out),
        .data_out        (data_out)
    );

    // --------------------------------------------------------------- clock gen
    initial begin
        clk = 0;
        forever #(CLK_PERIOD/2) clk = ~clk;
    end

    // ----------------------------------------- capture data_out when valid_out
    // Runs in parallel with the stimulus thread.
    // valid_out is asserted by the controller on every cycle a result row is
    // written (last_tile only), so we sample data_out at the same edge.
    always @(posedge clk) begin
        if (valid_out && (file != 0)) begin
            $fwrite(file, "Row %0d: ", row_capture);
            for (int col = 0; col < N_SIZE; col++) begin
                automatic logic signed [DATAWIDTH_output-1:0] val;
                val = data_out[col*DATAWIDTH_output +: DATAWIDTH_output];
                $fwrite(file, "%0d ", val);
            end
            $fwrite(file, "\n");
            row_capture = row_capture + 1;
        end
    end

    // ----------------------------------------------------------------- stimulus
    initial begin
        // ---- open output file ----
        file = $fopen("output_buffer.txt", "w");
        if (file == 0) begin
            $display("ERROR: Could not open output_buffer.txt");
            $finish;
        end
        row_capture = 0;

        // ---- initialise inputs ----
        in_A           = '0;
        weights        = '0;
        valid_in       = 0;
        load_weight    = 0;
        last_tile      = 0;
        first_iteration = 0;
        pp             = 0;
        rst_n          = 0;
        @(negedge clk);

        // ================================================================
        // PHASE 1 – load weights (all-ones)
        // ================================================================
        $display("[%0t] Loading ALL-ONES weights ...", $time);
        rst_n       = 1;
        load_weight = 1;

        for (int row = 0; row < N_SIZE; row++) begin
            weights = {32{8'sd1}};
            @(negedge clk);
        end

        load_weight = 0;
        @(negedge clk);
        $display("[%0t] Weights loaded.\n", $time);

        // ================================================================
        // TILE 1 – first_iteration=1, last_tile=0  (partial sums only)
        // ================================================================
        first_iteration = 1;
        last_tile       = 0;
        repeat(2) @(negedge clk);

        $display("[%0t] Tile 1 – sending %0d input rows ...", $time, num_of_raws);
        valid_in = 1;
        pp = 0;
        for (int row = 0; row < num_of_raws; row++) begin
            pp = pp+1;
            if (pp == 253) pp = 0;
            in_A = {32{pp}};  // All 32 inputs = pp (signed byte)
            @(negedge clk);
        end

        $display("[%0t] Tile 1 input done. Draining ...", $time);
        in_A = '0;
        while (!done) @(negedge clk);
        valid_in = 0;
        $display("[%0t] Tile 1 done.\n", $time);

        repeat(2) @(negedge clk);

        // ================================================================
        // TILE 2 – first_iteration=0, last_tile=0  (accumulate partials)
        // ================================================================
        first_iteration = 0;
        last_tile       = 0;
        valid_in        = 1;

        $display("[%0t] Tile 2 – sending %0d input rows ...", $time, num_of_raws);
        pp = 0;
        for (int row = 0; row < num_of_raws; row++) begin
            pp = pp+1;
            if (pp == 253) pp = 0;
            in_A = {32{pp}};  // All 32 inputs = pp (signed byte)
            @(negedge clk);
        end

        $display("[%0t] Tile 2 input done. Draining ...", $time);
        in_A = '0;
        while (!done) @(negedge clk);
        valid_in = 0;
        $display("[%0t] Tile 2 done.\n", $time);

        repeat(2) @(negedge clk);

        // ================================================================
        // TILE 3 – first_iteration=0, last_tile=1  (final – valid_out fires)
        // ================================================================
        first_iteration = 0;
        last_tile       = 1;
        valid_in        = 1;

        $display("[%0t] Tile 3 (last) – sending %0d input rows ...", $time, num_of_raws);
        pp = 0;
        for (int row = 0; row < num_of_raws; row++) begin
            pp = pp+1;
            if (pp == 253) pp = 0;
            in_A = {32{pp}};  // All 32 inputs = pp (signed byte)
            @(negedge clk);
        end

        $display("[%0t] Tile 3 input done. Draining ...", $time);
        in_A = '0;
        while (!done) @(negedge clk);
        valid_in = 0;
        $display("[%0t] Tile 3 done.\n", $time);

        // Give the capture thread a couple of extra cycles to finish writing
        // any outstanding valid_out pulses that arrive after done.
        repeat(4) @(negedge clk);

        // ----------------------------------------------------------------
        $display("[%0t] %0d rows captured. Closing file ...", $time, row_capture);
        $fclose(file);
        file = 0;
        $display("Results written to output_buffer.txt");
        $stop;
    end

endmodule