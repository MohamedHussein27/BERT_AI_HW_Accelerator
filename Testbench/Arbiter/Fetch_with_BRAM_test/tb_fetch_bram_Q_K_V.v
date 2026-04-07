`timescale 1ns/1ps

module tb_fetch_bram_Q_K_V;

    // ========================================================
    // Parameters
    // ========================================================
    parameter ADDR_WIDTH       = 16;
    parameter ORIGINAL_COLUMNS = 768;   // matrix columns before transpose
    parameter ORIGINAL_ROWS    = 512;   // matrix rows before transpose
    parameter NUM_BITS         = 8;     // quantized element width
    parameter DATA_WIDTH       = 256;   // bus width
    parameter CLK_PERIOD       = 10;
    
    // ========================================================
    // Buffer Select Codes (For Readability)
    // ========================================================
    localparam [3:0] BUF_Q = 4'b0011;
    localparam [3:0] BUF_K = 4'b0100;
    localparam [3:0] BUF_V = 4'b0101;

    // ========================================================
    // Signals
    // ========================================================
    reg clk, rst_n;
    
    // Fetch Logic Controls
    reg       start_fetch;
    reg       reset_in_addr_counter;
    reg       reset_wt_addr_counter;
    reg       Double_buffering;
    reg [3:0] Buffer_Select;
    reg       hold_addr_ptr;
    reg [1:0] Tiles_Control;
    
    // BRAM Port A (Write) Controls
    reg                  ena;
    reg                  wea;
    reg [ADDR_WIDTH-1:0] addra;
    reg [DATA_WIDTH-1:0] dina;

    // DUT Outputs
    wire                  fetch_done;
    wire                  busy;
    wire [DATA_WIDTH-1:0] doutb;
    wire [ADDR_WIDTH-1:0] addrb;
    
    // ========================================================
    // Clock Generation
    // ========================================================
    initial begin
        clk = 0;
        forever #(CLK_PERIOD/2) clk = ~clk;  // 100 MHz clock
    end

    // ========================================================
    // Device Under Test (DUT)
    // ========================================================
    fetch_bram_Q_K_V_top #(
        .ADDR_WIDTH       (ADDR_WIDTH),
        .ORIGINAL_COLUMNS (ORIGINAL_COLUMNS),
        .ORIGINAL_ROWS    (ORIGINAL_ROWS),
        .NUM_BITS         (NUM_BITS),
        .DATA_WIDTH       (DATA_WIDTH)
    ) u_top_1 (
        .clk                   (clk),
        .rst_n                 (rst_n),
        
        .start_fetch           (start_fetch),
        .reset_in_addr_counter (reset_in_addr_counter),
        .reset_wt_addr_counter (reset_wt_addr_counter),
        .Buffer_Select         (Buffer_Select),
        .Tiles_Control         (Tiles_Control),
        .Double_buffering      (Double_buffering),
        .hold_addr_ptr         (hold_addr_ptr),

        .wea                   (wea),
        .ena                   (ena),
        .addra                 (addra),
        .dina                  (dina),

        .fetch_done            (fetch_done),
        .doutb                 (doutb),
        .addrb                 (addrb),
        .busy                  (busy)
    );
    
    // ========================================================
    // Main Test Sequence
    // ========================================================
    integer i = 0;

    initial begin
        // 1. Initialize Signals
        rst_n                 = 0;
        start_fetch           = 0;
        reset_in_addr_counter = 0;
        reset_wt_addr_counter = 0;
        hold_addr_ptr         = 0;
        Double_buffering      = 1'b0;
        ena                   = 0;
        wea                   = 0;
        addra                 = 0;
        dina                  = 0;
        Buffer_Select         = BUF_K; // Default
        Tiles_Control         = 2'b01; // Tiling 32
        
        // 2. Apply Reset
        repeat(5) @(negedge clk);
        rst_n = 1;
        ena   = 1;
        repeat(2) @(negedge clk);

        // 3. Pre-load BRAM (Mocking the write_logic_gen)
        $display("\n========================================");
        $display("[TIME %0t] Writing BRAM with Test Data...", $time);
        $display("========================================");
        
        // 512*768*8 / 256 (bus width) = 12288 for each buffer 
        for (i = 0; i < 36864; i = i + 1) begin 
            wea   = 1;
            dina  = i * 2 + 2;      // deterministic pattern
            @(negedge clk);
            addra = addra + 1;      // increment write address
        end
        wea = 0;
        repeat(5) @(negedge clk);
        
        // 4. Execute Fetch Tests using Reusable Tasks
        $display("\n========================================");
        $display("[TIME %0t] Beginning Fetch Operations...", $time);
        $display("========================================");

        // Test 1: Fetch K buffer (Weights style, Tile 32)
        do_fetch(BUF_K, 2'b01, "K");

        // Test 2: Fetch Q buffer (Inputs style, Tile 512)
        do_fetch(BUF_Q, 2'b00, "Q");

        // Test 3: Fetch K buffer again (Continuing pointers)
        do_fetch(BUF_K, 2'b01, "K");

        // Test 4: Fetch Q buffer again (Continuing pointers)
        do_fetch(BUF_Q, 2'b00, "Q");

        // Test 5: Fetch V buffer (Resetting pointers first)
        reset_pointers(1, 1); // Reset both input and weight pointers
        do_fetch(BUF_V, 2'b01, "V");

        // End Simulation
        repeat(5) @(negedge clk);
        $display("\n[TIME %0t] All Tests Completed Successfully.", $time);
        $stop;
    end

    // ========================================================
    // Reusable Verification Tasks
    // ========================================================
    
    // Task to configure, trigger, and wait for a fetch operation
    task automatic do_fetch(
        input [3:0]  buf_sel,
        input [1:0]  tile_ctrl,
        input string buf_name
    );
        begin
            @(negedge clk);
            Buffer_Select = buf_sel;
            Tiles_Control = tile_ctrl;
            repeat(2) @(negedge clk); // Allow combinational logic to settle
            
            $display("   -> [TIME %0t] Starting fetch from %s buffer...", $time, buf_name);
            start_fetch = 1;
            @(negedge clk);
            start_fetch = 0;
            
            wait(fetch_done);
            $display("   -> [TIME %0t] Fetching %s buffer DONE.", $time, buf_name);
        end
    endtask

    // Task to cleanly reset the address pointers
    task automatic reset_pointers(
        input reset_in, 
        input reset_wt
    );
        begin
            @(negedge clk);
            $display("\n   -> [TIME %0t] Resetting Pointers (In:%0b, Wt:%0b)...", $time, reset_in, reset_wt);
            reset_in_addr_counter = reset_in;
            reset_wt_addr_counter = reset_wt;
            
            repeat(2) @(negedge clk);
            reset_in_addr_counter = 0;
            reset_wt_addr_counter = 0;
        end
    endtask

endmodule