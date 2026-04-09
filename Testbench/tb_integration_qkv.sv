`timescale 1ns / 1ps

module tb_integration_qkv();

    parameter FETCH_NUM_BUFFERS = 17;
    parameter WRITE_NUM_BUFFERS = 12;

    // Clock and Reset Generation
    reg clk;
    reg rst_n;
    
    initial begin
        clk = 0;
        forever #5 clk = ~clk; // 10ns period -> 100MHz clock
    end
    
    initial begin
        rst_n = 0;
        #20;
        rst_n = 1; // Release reset
    end

    // Top-Level Control Signals
    reg  start_inference;
    wire layer_done_wire;

    // =======================================================
    // Interconnect Wires (Refactored to _wire suffix)
    // =======================================================
    
    // Controller <-> Fetch Logic Signals
    wire       fetch_start_wire;
    wire [4:0] fetch_buffer_sel_wire; // 5-bit to match fetch_logic_gen
    wire [1:0] fetch_tiles_ctrl_wire;
    wire       fetch_double_buf_wire;
    wire       fetch_reset_wt_addr_counter_wire;
    wire       fetch_reset_in_addr_counter_wire;
    wire       fetch_hold_addr_ptr_wire;
    wire       fetch_stop_counting_wire;
    wire       fetch_busy_wire;
    wire       fetch_wt_done_wire;
    wire       fetch_in_done_wire;
    
    wire [10:0]                  fetch_bram_addr_wire;
    wire [FETCH_NUM_BUFFERS-1:0] fetch_bram_en_wire;

    // Handle width mismatch between 3-bit output from master controller and 4-bit input to fetch logic
    assign fetch_buffer_sel_wire[4] = 1'b0; 

    // Controller <-> Write Logic Signals
    wire [3:0] write_buffer_sel_wire;
    wire       write_start_wire;
    wire       write_double_buf_wire;
    wire       write_reset_addr_wire;
    
    wire       write_done_all_wire;
    wire       write_tile_done_wire;
    wire       write_busy_wire;
    
    wire [15:0]                  write_bram_addr_wire;
    wire [WRITE_NUM_BUFFERS-1:0] write_bram_we_wire;

    // Controller <-> Systolic Array Signals
    wire sa_valid_in_wire;
    wire sa_load_weight_wire;
    wire sa_first_iter_wire;
    wire sa_last_tile_wire;
    
    wire sa_done_wire;
    wire sa_valid_out_wire;
    wire sa_ready_wire;
    wire sa_busy_wire;
    wire [1023:0] sa_data_out_wire; // Assuming output matches width for N_SIZE=32, DATAWIDTH_out=32

    wire [255:0] doutb_wbi_wire;
    wire [255:0] doutb_qkv_wire;

    // --- ONE-CYCLE DELAY REGISTERS ---
    reg reg_sa_valid_in;
    reg reg_sa_valid_in_2;
    reg reg_sa_valid_in_3;
    reg reg_sa_load_weight;
    reg reg_sa_load_weight_2;
    reg reg_sa_first_iter;
    reg reg_sa_last_tile;

    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            reg_sa_valid_in    <= 1'b0;
            reg_sa_load_weight <= 1'b0;
            reg_sa_first_iter  <= 1'b0;
            reg_sa_last_tile   <= 1'b0;
        end else begin
            reg_sa_valid_in      <= sa_valid_in_wire;
            reg_sa_load_weight   <= sa_load_weight_wire;
            reg_sa_first_iter    <= sa_first_iter_wire;
            reg_sa_last_tile     <= sa_last_tile_wire;
            reg_sa_load_weight_2 <= reg_sa_load_weight;
            reg_sa_valid_in_2    <= reg_sa_valid_in;
            reg_sa_valid_in_3    <= reg_sa_valid_in_2;
        end
    end

    // Testbench Write & Read Signals
    // For W_B_I preload
    reg [10:0]  tb_wbi_addr;
    reg [255:0] tb_wbi_data;
    reg         tb_wbi_we;

    // For Q_K_V Verification Read
    reg         tb_read_mode; // 1 = TB controls QKV read port. 0 = Hardware controls it.
    reg [15:0]  tb_qkv_addrb;
    reg         tb_qkv_enb;


    // =======================================================
    // Module Instantiations
    // =======================================================

    // --- Main Controller ---
    transformer_master_ctrl u_cu (
        .clk                         (clk),
        .rst_n                       (rst_n),
        .start_inference             (start_inference),
        .layer_done                  (layer_done_wire),

        .fetch_start                 (fetch_start_wire),
        .fetch_buffer_sel            (fetch_buffer_sel_wire[3:0]), // Master Outputs 4 bits
        .fetch_tiles_ctrl            (fetch_tiles_ctrl_wire),
        .fetch_double_buf            (fetch_double_buf_wire),
        .fetch_hold_addr_ptr         (fetch_hold_addr_ptr_wire),
        .fetch_reset_in_addr_counter (fetch_reset_in_addr_counter_wire), 
        .fetch_reset_wt_addr_counter (fetch_reset_wt_addr_counter_wire),
        .fetch_stop_counting         (fetch_stop_counting_wire),
        .fetch_in_done               (fetch_in_done_wire),
        .fetch_wt_done               (fetch_wt_done_wire),
        .fetch_busy                  (fetch_busy_wire), 

        .write_buffer_sel            (write_buffer_sel_wire),
        .write_start                 (write_start_wire),
        .write_double_buf            (write_double_buf_wire),
        .write_reset_address_counter (write_reset_addr_wire),
        .write_done_all              (write_done_all_wire),
        .write_tile_done             (write_tile_done_wire),
        .write_busy                  (write_busy_wire),

        .sa_valid_in                 (sa_valid_in_wire),
        .sa_load_weight              (sa_load_weight_wire),
        .sa_first_iter               (sa_first_iter_wire),
        .sa_last_tile                (sa_last_tile_wire),
        .sa_done                     (sa_done_wire),
        .sa_valid_out                (sa_valid_out_wire)
    );

    // --- Fetch Logic ---
    fetch_logic_gen u_fetch_logic (
        .clk                   (clk),
        .rst_n                 (rst_n),
        .start_fetch           (fetch_start_wire),
        .reset_in_addr_counter (fetch_reset_in_addr_counter_wire), 
        .reset_wt_addr_counter (fetch_reset_wt_addr_counter_wire),
        .Buffer_Select         (fetch_buffer_sel_wire),
        .Tiles_Control         (fetch_tiles_ctrl_wire),
        .Double_buffering      (fetch_double_buf_wire),
        .hold_addr_ptr         (fetch_hold_addr_ptr_wire),
        .stop_counting         (fetch_stop_counting_wire),
        
        .bram_addr             (fetch_bram_addr_wire),
        .bram_en               (fetch_bram_en_wire),
        .fetch_wt_done         (fetch_wt_done_wire),
        .fetch_in_done         (fetch_in_done_wire),
        .busy                  (fetch_busy_wire)
    );

    // --- Write Logic ---
    write_logic_gen u_write_logic (
        .clk                (clk),
        .rst_n              (rst_n),
        .start_write        (write_start_wire),
        .reset_addr_counter (write_reset_addr_wire),
        .Buffer_Select      (write_buffer_sel_wire),
        .Double_buffering   (write_double_buf_wire),
        .sipo_valid_out     (1'b0), // Tied low for QKV processing
        .sipo_mode          (1'b0), // Tied low for QKV processing

        .bram_addr          (write_bram_addr_wire),
        .bram_we            (write_bram_we_wire),
        .write_all_done     (write_done_all_wire),
        .write_tile_done    (write_tile_done_wire),
        .busy               (write_busy_wire)
    );

    // --- Q_K_V BRAM ---
    Q_K_V_buffer u_qkv_buffer (
        // Port A (Write) - Extracting bits 0 (Q), 1 (K), and 2 (V) from the write enable bus
        .clka   (clk),
        .ena    (write_bram_we_wire[0] | write_bram_we_wire[1] | write_bram_we_wire[2]),       
        .wea    (write_bram_we_wire[0] | write_bram_we_wire[1] | write_bram_we_wire[2]),       
        .addra  (write_bram_addr_wire),
        .dina   (sa_data_out_wire[255:0]), 
        
        // Port B (Read) - Multiplexed between Testbench Read and Fetch Logic bits 3, 4, 5 (Q, K, V read)
        .clkb   (clk),
        .enb    (tb_read_mode ? tb_qkv_enb   : (fetch_bram_en_wire[3] | fetch_bram_en_wire[4] | fetch_bram_en_wire[5])), 
        .addrb  (tb_read_mode ? tb_qkv_addrb : fetch_bram_addr_wire), 
        .doutb  (doutb_qkv_wire)       
    );

    // --- W_B_I BRAM ---
    W_B_I_Buffer u_wbi_buffer (
        // Port A (Write) - Controlled by TB 
        .clka   (clk),
        .ena    (tb_wbi_we),
        .wea    (tb_wbi_we),
        .addra  (tb_wbi_addr),
        .dina   (tb_wbi_data),
        
        // Port B (Read) - Extracting bits 0 (W), 1 (B), and 2 (I) from the fetch enable bus
        .clkb   (clk),
        .enb    (fetch_bram_en_wire[0] | fetch_bram_en_wire[1] | fetch_bram_en_wire[2]),
        .addrb  (fetch_bram_addr_wire),
        .doutb  (doutb_wbi_wire)
    );

    // --- Systolic Array ---
    systolic_top u_systolic (
        .clk             (clk),
        .rst_n           (rst_n),
        .in_A            (doutb_qkv_wire), 
        .weights         (doutb_wbi_wire), 
        
        .valid_in        (reg_sa_valid_in_3),
        .load_weight     (reg_sa_load_weight_2),
        .first_iteration (reg_sa_first_iter),
        .last_tile       (reg_sa_last_tile),
        
        .ready           (sa_ready_wire),
        .busy            (sa_busy_wire),
        .done            (sa_done_wire),
        .valid_out       (sa_valid_out_wire),
        .data_out        (sa_data_out_wire)
    );

    // =======+++++++++++++++++++++++++++++++++++++++++++++++++++==================================================================
    // Test Stimulus Sequence & Tasks
    // =======+++++++++++++++++++++++++++++++++++++++++++++++++++==================================================================
    integer i;
    integer row; // Used for initializing the matrices
    logic[7:0] k;

    reg [255:0] matrix_32x32 [0:31]; 
    reg [255:0] matrix_512x32 [0:511];
    reg [255:0] output_matrix_512x32_Q [0:511]; // Matrix to store read-back Q/K/V tile
    reg [255:0] output_matrix_512x32_K [0:511];
    reg [255:0] output_matrix_512x32_V [0:511];

    // -------------------------------------------------------------------------
    // TASK: Write Weight
    // -------------------------------------------------------------------------
    task automatic write_weight(
        input [15:0] start_addr,
        ref reg [255:0] data_matrix [0:31]
    );
        integer w_idx;
        begin
            $display("Task: Loading 32x32 Weight Matrix at address %0d...", start_addr);
            tb_wbi_we = 1;
            for (w_idx = 0; w_idx < 32; w_idx = w_idx + 1) begin
                @(posedge clk);
                tb_wbi_addr = start_addr + w_idx;
                tb_wbi_data = data_matrix[w_idx]; 
            end
            @(posedge clk);
            tb_wbi_we = 0; 
        end
    endtask

    // -------------------------------------------------------------------------
    // TASK: Load Input
    // -------------------------------------------------------------------------
    task automatic load_input(
        input [15:0] start_addr,
        ref reg [255:0] data_matrix [0:511]
    );
        integer in_idx;
        begin
            $display("Task: Loading 512x32 Input Matrix at address %0d...", start_addr);
            tb_wbi_we = 1;
            for (in_idx = 0; in_idx < 512; in_idx = in_idx + 1) begin
                @(posedge clk);
                tb_wbi_addr = start_addr + in_idx;
                tb_wbi_data = data_matrix[in_idx]; 
            end
            @(posedge clk);
            tb_wbi_we = 0; 
        end
    endtask

    // -------------------------------------------------------------------------
    // MAIN INITIAL BLOCK
    // -------------------------------------------------------------------------
    initial begin
        // 1. Initial State
        start_inference = 0;
        
        // TB Write signals
        tb_wbi_we       = 0;
        tb_wbi_addr     = 0;
        tb_wbi_data     = 256'd0;
        
        // TB Read signals
        tb_read_mode    = 0;
        tb_qkv_addrb    = 0;
        tb_qkv_enb      = 0;

        // Populate Matrix Definitions
        for (row = 0; row < 32; row = row + 1) begin
            k = row; 
           matrix_32x32[row] = {32{k}};
        end
        for (row = 0; row < 512; row = row + 1) begin
            k = row;
            if (k >= 256) begin
                k = 6;
            end
            matrix_512x32[row] = {32{k+2}};
            
        end
        
        // Load Arrays
        write_weight(16'd0, matrix_32x32);
        write_weight(16'd32, matrix_32x32); // db
        load_input(16'd112, matrix_512x32);  // db
        load_input(16'd624, matrix_512x32);

        // the controller will start working here 
        @(posedge clk);
        start_inference = 1;
        @(posedge clk);
        start_inference = 0;

        // Give the controller a few clock cycles to transition OUT of IDLE
        repeat(5) @(posedge clk);

        // Wait until the controller returns to the IDLE state (assuming IDLE is 2'd0)
        wait(u_cu.state == 2'd0);  
        
        dump_qkv_buffer_to_file("qkv_computed_results.txt");
        $finish;
    end

    // -------------------------------------------------------------------------
    // TASK: Dump Q_K_V Buffer to File
    // -------------------------------------------------------------------------
    task automatic dump_qkv_buffer_to_file(
        input string filename
    );
        integer fd;
        integer r_idx;
        begin
            $display("Task: Dumping entire Q_K_V Buffer to %s...", filename);
            
            // 0. Open the file for writing
            fd = $fopen(filename, "w");
            if (fd == 0) begin
                $display("ERROR: Could not open file %s", filename);
                $finish;
            end
            
            // 1. Take control of the BRAM Read Port
            tb_read_mode = 1'b1; 
            tb_qkv_enb   = 1'b1;
            
            // 2. Loop MAX_DEPTH + 1 times (36864 + 1) due to 1-cycle read latency
            for (r_idx = 0; r_idx <= 36864; r_idx = r_idx + 1) begin
                
                // Drive the address (for 0 to 36863)
                if (r_idx < 36864) begin
                    tb_qkv_addrb = r_idx;
                end
                
                // Wait for the clock edge
                @(posedge clk);
                
                // Capture the data.
                if (r_idx > 0) begin
                    #1; // Delay to ensure BRAM data is stable after clock edge
                    $fdisplay(fd, "%h", doutb_qkv_wire);
                end
            end
            
            // 3. Release control and close the file
            tb_qkv_enb   = 1'b0;
            tb_read_mode = 1'b0; 
            $fclose(fd);
            
            $display("Task: Dump Complete. File saved to %s", filename);
        end
    endtask




// =======================================================
    // Monitor: Dump Systolic Array Weights
    
    // 1. Create a normal 2D array in the testbench to hold the probed values
    wire [7:0] probed_pe_weights [0:31][0:31];

    // 2. Use a genvar loop to map the internal PE weights to our testbench array.
    // Since genvars are evaluated at compile-time, this is perfectly legal!
    genvar gr, gc;
    generate
        for (gr = 0; gr < 32; gr = gr + 1) begin : probe_row
            for (gc = 0; gc < 32; gc = gc + 1) begin : probe_col
                // Map the internal PE weight register to the TB array
                assign probed_pe_weights[gr][gc] = u_systolic.u_systolic.row_loop[gr].col_loop[gc].pe_inst.weight;
            end
        end
    endgenerate

    // 3. Print the array using standard variables
    always_comb begin
        $display("\n==================================================");
        $display("   SYSTOLIC ARRAY WEIGHT REGISTERS LATCHED");
        $display("==================================================");
        
        // Loop through our mapped testbench array (variable indexing is legal here!)
        // for (int r = 0; r < 32; r++) begin
        //     for (int c = 0; c < 32; c++) begin
        //         $display("PE[%0d][%0d] Weight = %h", r, c, probed_pe_weights[r][c]);
        //     end
        // end

         $display("PE[0][0] Weight = %h", u_systolic.u_systolic.row_loop[0].col_loop[0].pe_inst.weight); // 0
         $display("PE[1][0] Weight = %h", u_systolic.u_systolic.row_loop[1].col_loop[0].pe_inst.weight); // 0
         $display("PE[2][0] Weight = %h", u_systolic.u_systolic.row_loop[2].col_loop[0].pe_inst.weight); // 0
         $display("PE[31][0] Weight = %h", u_systolic.u_systolic.row_loop[31].col_loop[0].pe_inst.weight); // 31 
        $display("==================================================\n");
    end
endmodule