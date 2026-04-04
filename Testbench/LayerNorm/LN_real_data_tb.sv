`timescale 1ns / 1ps

module tb_layernorm_real();

    // --------------------------------------------------------
    // Parameters & Precision
    // --------------------------------------------------------
    localparam int DATA_WIDTH = 32;
    localparam int PE_COUNT   = 32;
    localparam int ROWS       = 128;
    localparam int COLS       = 768;
    localparam int CHUNKS     = COLS / PE_COUNT; // 24 chunks per row
    localparam int TOTAL_DATA = ROWS * COLS;     // 98,304 elements

    // Float comparison parameters (Matching original golden TB)
    localparam real Q_SCALE = 67108864.0; // 2^26 
    localparam real TOLERANCE_FLOAT = 0.05; 

    // --------------------------------------------------------
    // DUT Signals 
    // --------------------------------------------------------
    logic clk;
    logic rst_n;
    logic data_valid;
    
    logic signed [DATA_WIDTH-1:0] buffer_rdata  [0:PE_COUNT-1];
    logic signed [DATA_WIDTH-1:0] norm_out_data [0:PE_COUNT-1];
    
    logic norm_out_valid;
    logic done;
    logic busy;

    // --------------------------------------------------------
    // File I/O Memories
    // --------------------------------------------------------
    logic signed [DATA_WIDTH-1:0] mem_inputs   [0:TOTAL_DATA-1];
    logic signed [DATA_WIDTH-1:0] mem_expected [0:TOTAL_DATA-1];
    logic signed [DATA_WIDTH-1:0] mem_gamma    [0:COLS-1];
    logic signed [DATA_WIDTH-1:0] mem_beta     [0:COLS-1];

    logic signed [DATA_WIDTH-1:0] expected_val;
    logic signed [DATA_WIDTH-1:0] actual_val;

    // Real values for comparison
    real real_expected;
    real real_actual;
    real real_diff;

    // Tracking variables
    int errors = 0;
    int matche_exp = 0;
    int check_row = 0;
    int check_chunk = 0;
    int global_idx = 0;

    // --------------------------------------------------------
    // Instantiate DUT 
    // --------------------------------------------------------
    layernorm_top #(
        .DATAWIDTH(DATA_WIDTH)
    ) uut (
        .clk(clk),
        .rst_n(rst_n),
        .data_valid(data_valid),
        .buffer_rdata(buffer_rdata),
        .norm_out_data(norm_out_data),
        .norm_out_valid(norm_out_valid),
        .done(done),
        .busy(busy)
    );

    // --------------------------------------------------------
    // Clock Generation
    // --------------------------------------------------------
    always #5 clk = ~clk; // 100MHz clock

    // --------------------------------------------------------
    // Main Stimulus Process
    // --------------------------------------------------------
    initial begin
        // 1. Initialize Memories from Hex Files
        $display("==================================================");
        $display("   LOADING Q5.26 HEX FILES...");
        $display("==================================================");
        $readmemh("inputs.hex", mem_inputs);
        $readmemh("expected.hex", mem_expected);
        $readmemh("real_gamma.hex", mem_gamma);
        $readmemh("real_beta.hex", mem_beta);
        $display("Files loaded successfully.");

        // 2. Reset Sequence
        clk = 0;
        rst_n = 0;
        data_valid = 0;
        for (int i=0; i<PE_COUNT; i++) begin
            buffer_rdata[i] = 0;
        end
        
        #20 rst_n = 1;
        #10;

        $display("==================================================");
        $display("   STARTING DATA STREAMING (128 Rows x 768 Cols)");
        $display("==================================================");

        // 3. Stream Data Row by Row using the 3-Pass Method
        for (int r = 0; r < ROWS; r++) begin
            
            // --- PASS 1: MEAN COMPUTATION ---
            for (int c = 0; c < CHUNKS; c++) begin
                @(negedge clk);
                data_valid = 1;
                for (int p = 0; p < PE_COUNT; p++) begin
                    buffer_rdata[p] = mem_inputs[r*COLS + c*PE_COUNT + p];
                end
            end
            @(negedge clk) data_valid = 0;

            #20;

            // --- PASS 2: VARIANCE COMPUTATION ---
            for (int c = 0; c < CHUNKS; c++) begin
                @(negedge clk);
                data_valid = 1;
                for (int p = 0; p < PE_COUNT; p++) begin
                    buffer_rdata[p] = mem_inputs[r*COLS + c*PE_COUNT + p];
                end
            end
            @(negedge clk) data_valid = 0;

            // --- WAIT FOR SQRT & INJECT GAMMA/BETA ---
            wait(uut.u_fsm.state == 3'd4); // ST_CALC_SQRT
            
            @(negedge clk);
            data_valid = 1'b1; 
            for (int p = 0; p < PE_COUNT; p++) buffer_rdata[p] = mem_gamma[p];
            
            @(negedge clk);
            for (int p = 0; p < PE_COUNT; p++) buffer_rdata[p] = mem_beta[p];
            
            @(negedge clk);
            data_valid = 1'b0;
            for (int p = 0; p < PE_COUNT; p++) buffer_rdata[p] = '0;

            // --- PASS 3: NORMALIZATION & AFFINE ---
            wait(uut.u_fsm.state == 3'd5); // ST_PASS3_NORM
            for (int c = 0; c < CHUNKS; c++) begin
                @(negedge clk);
                data_valid = 1;
                for (int p = 0; p < PE_COUNT; p++) begin
                    buffer_rdata[p] = mem_inputs[r*COLS + c*PE_COUNT + p];
                end
            end
            @(negedge clk) data_valid = 0;

            wait(done == 1'b1);
            @(negedge clk);
        end

        // Wait for final checks to complete
        #100;
        $display("==================================================");
        $display("   SIMULATION COMPLETE");
        $display("   Total Matches : %0d", matche_exp);
        $display("   Total Errors  : %0d", errors);
        $display("==================================================");
        $finish;
    end

    // --------------------------------------------------------
    // Monitor and Checking Process
    // --------------------------------------------------------
    always @(negedge clk) begin
        if (rst_n && norm_out_valid) begin
            for (int p = 0; p < PE_COUNT; p++) begin
                
                global_idx = check_row * COLS + check_chunk * PE_COUNT + p;
                expected_val = mem_expected[global_idx];
                actual_val = norm_out_data[p];
                
                // Convert to real format exactly like the golden TB
                real_expected = real'(expected_val) / Q_SCALE;
                real_actual   = real'(actual_val) / Q_SCALE;
                
                // Calculate absolute difference in float domain
                real_diff = real_actual - real_expected;
                if (real_diff < 0) real_diff = -real_diff;
                
                // Compare against float tolerance (0.05)
                if (real_diff > TOLERANCE_FLOAT) begin
                    $error("[ERROR] Row %0d, Col %0d | Expected: %f, Got: %f | Diff: %f", 
                             check_row, (check_chunk*PE_COUNT + p), real_expected, real_actual, real_diff);
                    errors++;
                end else begin
                    matche_exp++;
                end
            end
            
            // Increment Chunk and Row counters
            check_chunk++;
            if (check_chunk == CHUNKS) begin
                check_chunk = 0;
                $display("Successfully verified output for Row %0d", check_row);
                check_row++;
            end
        end
    end

endmodule