`timescale 1ns / 1ps

module tb_controller_sm();

    parameter FETCH_NUM_BUFFERS = 17;
    parameter WRITE_NUM_BUFFERS = 12;

    // Clock and Reset Generation
    reg clk;
    reg rst_n;

    initial begin
        clk = 0;
        forever #5 clk = ~clk;
        // 10ns period -> 100MHz clock
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
    wire [15:0]                  fetch_bram_addr_wire;
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

   
    // Assuming output matches width for N_SIZE=32, DATAWIDTH_out=32

    wire [255:0] doutb_wbi_wire;
    wire [255:0] doutb_qkv_wire;
    wire [255:0] doutb_qkt_wire;

    // =======================================================
    // NEW PIPELINE WIRES (Softmax, Quantize, PISO/SIPO)
    // =======================================================
    wire        write_sipo_mode_wire;
    wire        quantize_valid_in_wire;
    wire [7:0]  quantize_param_addr_wire;
    wire        quantize_u_valid_in_wire;
    wire        p2s_valid_out_wire;
    wire        p2s_busy_wire;
    
    wire        softmax_start_wire;
    wire        softmax_valid_in_wire;
    wire        softmax_done_wire;
    wire        softmax_out_valid_wire;
    wire        softmax_out_last_wire;
    wire        softmax_busy_wire;
    wire        softmax_in_ready_wire;
    
    wire        s2p_out_valid_wire;
    
    wire [31:0] m0_out_wire;
    wire [7:0] s_out_wire;
    
    wire [31:0][31:0] quantize_piso_data_wire;
    wire         quantize_piso_valid_wire;
    
    wire [31:0] p2s_out_wire;
    wire [15:0] softmax_out_wire;
    
    wire [7:0]  quantized_sm_out_wire;
    wire        quantized_sm_valid_out_wire;
    
    wire [255:0] s2p_out_wire;

    // --- ONE-CYCLE DELAY REGISTERS ---
    reg reg_sa_valid_in;
    reg reg_sa_valid_in_2;
    reg reg_sa_valid_in_3;
    reg reg_sa_load_weight;
    reg reg_sa_load_weight_2;
    reg reg_sa_first_iter;
    reg reg_sa_last_tile;

    /*always @(posedge clk or negedge rst_n) begin
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
    end*/

    logic quantize_valid_in_wire_reg;  
    logic quantize_valid_in_wire_reg_2;
    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            quantize_valid_in_wire_reg  <= 1'b0;
        end else begin
            quantize_valid_in_wire_reg   <= quantize_valid_in_wire;
            quantize_valid_in_wire_reg_2 <= quantize_valid_in_wire_reg;
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
    
    // For Q_Kt
    reg         tb_qkt_we;
    reg         tb_qkt_enb;        
    reg [12:0]  tb_qkt_addra;
    reg [12:0]  tb_qkt_addrb;      
    reg [255:0] tb_qkt_data;

    // For SM Buffer
    reg         tb_sm_enb;
    reg [12:0]  tb_sm_addrb;
    wire [255:0] tb_sm_buffer_out;

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
        .fetch_buffer_sel            (fetch_buffer_sel_wire), // Master Outputs 4 bits
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
    
        .write_sipo_mode             (write_sipo_mode_wire),

        .quantize_valid_in           (quantize_valid_in_wire),          // Vector Quantization
        .quantize_param_addr         (quantize_param_addr_wire),
        .quantize_u_valid_in         (quantize_u_valid_in_wire),

        .piso_valid_out              (p2s_valid_out_wire),
        .piso_busy                   (p2s_busy_wire),

        .softmax_start               (softmax_start_wire),
        .softmax_valid_in            (softmax_valid_in_wire), // Fixed missing comma
        .softmax_done                (softmax_done_wire),
        .softmax_out_valid           (softmax_out_valid_wire), // Added: Used in WAIT_SOFTMAX
        .softmax_out_last            (softmax_out_last_wire),
        .softmax_busy                (softmax_busy_wire),
        .softmax_in_ready            (softmax_in_ready_wire)
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
        .sipo_valid_out     (s2p_out_valid_wire),           // Tied to SIPO valid
        .sipo_mode          (write_sipo_mode_wire),         // Controlled by CU

        .bram_addr          (write_bram_addr_wire),
        .bram_we            (write_bram_we_wire),
        .write_all_done     (write_done_all_wire),
        .write_tile_done    (write_tile_done_wire),
        .busy               (write_busy_wire)
    );

    // --- Q_kt_Buffer buffer
    Q_Kt_buffer u_qkt_buffer (
        // Port A: 256-bit Write Port
        .clka(clk),
        .ena(tb_qkt_we),
        .wea(tb_qkt_we),
        .addra(tb_qkt_addra),
        .dina(tb_qkt_data),

        // Port B: 256-bit Read Port 
        .clkb(clk),
        .enb(tb_read_mode ? tb_qkt_enb   :  fetch_bram_en_wire[6]),
        .addrb(tb_read_mode ? tb_qkt_addrb : fetch_bram_addr_wire[12:0]),
        .doutb(doutb_qkt_wire)
    );

    scale_rom scale_rom_u (
        .addr(quantize_param_addr_wire), // Fixed missing comma
        .m0_out(m0_out_wire),
        .s_out(s_out_wire)
    );

    vector_quantize #(
        .VEC_SIZE     (32),
        .DATAWIDTH_in (8),
        .DATAWIDTH_out(32),
        .M_width      (32),
        .S_width      (8)
        ) de_quantize_to_piso (
        .clk(clk),
        .rst_n(rst_n),
        .valid_in(quantize_valid_in_wire),
        .data_in(doutb_qkt_wire),
        .scale_M(m0_out_wire),
        .scale_S(s_out_wire),
        .data_out(quantize_piso_data_wire),
        .valid_out(quantize_piso_valid_wire)
    );

    parrallel2serial p2s (
        .clk(clk),
        .rst_n(rst_n),
        .valid_in(quantize_piso_valid_wire),
        .data_in(quantize_piso_data_wire),
        .data_out(p2s_out_wire),
        .busy(p2s_busy_wire),
        .valid_out(p2s_valid_out_wire)
    );

    bert_softmax u_softmax(
        .clk(clk),
        .rst_n(rst_n),
        .start(softmax_start_wire),             // Pulse to begin
        .vec_len_cfg(10'd512),                    // Runtime vector length
        .in_valid(p2s_valid_out_wire),
        .in_ready(softmax_in_ready_wire),
        .in_data(p2s_out_wire),                 // Q5.26 signed
        .out_valid(softmax_out_valid_wire),
        .out_data(softmax_out_wire),            // Q1.15 unsigned
        .out_last(softmax_out_last_wire),       // Pulses on last output
        .busy(softmax_busy_wire),
        .done(softmax_done_wire)
    );

    // Quantize SM Output
    quantize #(
        .DATAWIDTH_in (16),
        .DATAWIDTH_out(8),
        .M_width      (32),
        .S_width      (8)    
        )
        u_quantize 
        (
        .clk(clk),
        .rst_n(rst_n),
        .valid_in(quantize_u_valid_in_wire),
        .data_in(softmax_out_wire),
        .scale_M(m0_out_wire),
        .scale_S(s_out_wire),
        .data_out(quantized_sm_out_wire),
        .valid_out(quantized_sm_valid_out_wire)
    );

    // Serial To Parallel
    serial2parrallel u_sipo(
        .clk(clk),
        .rst_n(rst_n),
        .valid_in(softmax_out_valid_wire),      // Assuming mapped logic is correct
        .data_in(quantized_sm_out_wire),
        .data_out(s2p_out_wire),
        .valid_out(s2p_out_valid_wire)
    );

    // SM Buffer
    SM_buffer u_sm_buffer (                     // Fixed missing instance name
        .clka(clk),
        .ena(write_bram_we_wire[4]),
        .wea(write_bram_we_wire[4]),
        .addra(write_bram_addr_wire[12:0]),
        .dina(s2p_out_wire),
        .clkb(clk),
        .enb(tb_read_mode ? tb_sm_enb : fetch_bram_en_wire[7]),
        .addrb(tb_read_mode ? tb_sm_addrb : fetch_bram_addr_wire[12:0]),
        .doutb(tb_sm_buffer_out)
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
    reg [255:0] matrix_512x512_QKt [0:8191];

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
        
        // Initialize QKt Signals
        tb_qkt_we       = 0;
        tb_qkt_addra    = 0; // Fixed undeclared variable (changed from tb_qkt_addr)
        tb_qkt_addrb    = 0; 
        tb_qkt_data     = 0;
        tb_qkt_enb      = 0;
        
        // Initialize SM Signals
        tb_sm_enb       = 0;
        tb_sm_addrb     = 0;

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
        matrix_512x512_QKt[0] = {32{8'd3}};
        matrix_512x512_QKt[1] = {32{8'd4}};
        matrix_512x512_QKt[8191] = {32{8'd7}};
        for (row = 2; row < 8191; row = row + 1) begin
           matrix_512x512_QKt[row] = {32{8'd5}};
        end

        // Load Arrays
        //write_weight(16'd0, matrix_32x32);
        //write_weight(16'd32, matrix_32x32); // db
        //load_input(16'd112, matrix_512x32);
        // db
        //load_input(16'd624, matrix_512x32);

        load_qkt(13'd0, matrix_512x512_QKt);
        
        // the controller will start working here 
        @(posedge clk);
        start_inference = 1;
        @(posedge clk);
        start_inference = 0;

        // Give the controller a few clock cycles to transition OUT of IDLE
        repeat(5) @(posedge clk);
        
        // Wait until the controller returns to the IDLE state (assuming IDLE is 2'd0)
        wait(u_cu.softmax_row_cnt == 1);
        dump_qkv_buffer_to_file("qkv_computed_results.txt");
        $finish;
    end

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
    // TASK: Load QKt Buffer
    // -------------------------------------------------------------------------
    task automatic load_qkt(
        input [12:0] start_addr,
        ref reg [255:0] data_matrix [0:8191]
    );
        integer in_idx;
        begin
            $display("Task: Loading 512x512 QKt Matrix at address %0d...", start_addr);
            tb_qkt_we = 1;
            
            for (in_idx = 0; in_idx < 8192; in_idx = in_idx + 1) begin
                @(posedge clk);
                tb_qkt_addra = start_addr + in_idx;
                tb_qkt_data = data_matrix[in_idx]; 
            end

            
            @(posedge clk);
            tb_qkt_we = 0; 
        end
    endtask

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
                    #1;
                    // Delay to ensure BRAM data is stable after clock edge
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

    

    // 3. Print the array using standard variables
    always_comb begin
        $display("\n==================================================");
        $display("   SYSTOLIC ARRAY WEIGHT REGISTERS LATCHED");
        $display("==================================================");
        
        /*$display("tile_counter = %0d", u_cu.tile_done_counter);
        $display("PE[0][0] Weight = %h", u_systolic.u_systolic.row_loop[0].col_loop[0].pe_inst.weight); // 0
        $display("PE[1][0] Weight = %h", u_systolic.u_systolic.row_loop[1].col_loop[0].pe_inst.weight); // 0
        $display("PE[2][0] Weight = %h", u_systolic.u_systolic.row_loop[2].col_loop[0].pe_inst.weight); // 0
        $display("PE[31][0] Weight = %h", u_systolic.u_systolic.row_loop[31].col_loop[0].pe_inst.weight); // 31 
        $display("==================================================\n");*/
    end


endmodule