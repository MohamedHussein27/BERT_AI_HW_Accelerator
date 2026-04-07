`timescale 1ns / 1ps

module tb_transformer_top();

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
    wire layer_done;

    // Controller <-> Fetch Logic Signal
    wire       fetch_start;
    wire [3:0] fetch_buffer_sel;
    wire [1:0] fetch_tiles_ctrl;
    wire       fetch_double_buf;
    wire       fetch_reset_addr;
    wire       hold_addr_ptr;
    
    wire       fetch_done;
    
    wire [15:0] fetch_bram_addr;
    wire        fetch_bram_en;

    wire [255:0] doutb_wbi;
    wire [255:0] doutb_qkv;

    // Controller <-> Write Logic Signals
    wire [3:0] write_buffer_sel;
    wire       write_start;
    wire       write_double_buf;
    wire       write_reset_addr;
    
    wire       write_done;
    wire       write_tile_done;
    wire       write_busy;
    
    wire [15:0] write_bram_addr;
    wire        write_bram_we;

    // Controller <-> Systolic Array Signals
    wire cu_sa_valid_in;
    wire cu_sa_load_weight;
    wire cu_sa_first_iter;
    wire cu_sa_last_tile;
    
    wire sa_done;
    wire sa_valid_out;
    wire sa_ready;
    wire sa_busy;
    wire [1023:0] sa_data_out; // Assuming output matches width for N_SIZE=32, DATAWIDTH_out=32

    // --- ONE-CYCLE DELAY REGISTERS ---
    reg reg_sa_valid_in;
    reg reg_sa_load_weight;
    reg reg_sa_first_iter;
    reg reg_sa_last_tile;

    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            reg_sa_valid_in    <= 1'b0;
            reg_sa_load_weight <= 1'b0;
            reg_sa_first_iter  <= 1'b0;
            reg_sa_last_tile   <= 1'b0;
        end else begin
            reg_sa_valid_in    <= cu_sa_valid_in;
            reg_sa_load_weight <= cu_sa_load_weight;
            reg_sa_first_iter  <= cu_sa_first_iter;
            reg_sa_last_tile   <= cu_sa_last_tile;
        end
    end


    // Testbench Write & Read Signals
    // For W_B_I preload
    reg [15:0]  tb_wbi_addr;
    reg [255:0] tb_wbi_data;
    reg         tb_wbi_we;

    // For Q_K_V Verification Read
    reg         tb_read_mode; // 1 = TB controls QKV read port. 0 = Hardware controls it.
    reg [15:0]  tb_qkv_addrb;
    reg         tb_qkv_enb;

    // Module Instantiations
    // --- Main Controller ---
    transformer_master_ctrl u_cu (
        .clk                         (clk),
        .rst_n                       (rst_n),
        .start_inference             (start_inference),
        .layer_done                  (layer_done),

        .fetch_start                 (fetch_start),
        .fetch_buffer_sel            (fetch_buffer_sel),
        .fetch_tiles_ctrl            (fetch_tiles_ctrl),
        .fetch_double_buf            (fetch_double_buf),
        .fetch_reset_address_counter (fetch_reset_addr),
        .hold_addr_ptr               (hold_addr_ptr),
        .fetch_done                  (fetch_done),
        .fetch_busy                  (1'b0), // Tied off

        .write_buffer_sel            (write_buffer_sel),
        .write_start                 (write_start),
        .write_double_buf            (write_double_buf),
        .write_reset_address_counter (write_reset_addr),
        .write_done                  (write_done),
        .write_tile_done             (write_tile_done),
        .write_busy                  (write_busy),

        .sa_valid_in                 (cu_sa_valid_in),
        .sa_load_weight              (cu_sa_load_weight),
        .sa_first_iter               (cu_sa_first_iter),
        .sa_last_tile                (cu_sa_last_tile),
        .sa_done                     (sa_done),
        .sa_valid_out                (sa_valid_out),

        // Softmax and LayerNorm Ignored for this test
        .softmax_start               (), 
        .softmax_done                (1'b0),
        .ln_valid_out                (1'b0),
        .ln_done                     (1'b0),
        .ln_valid_in                 ()
    );

    // --- Fetch Logic ---
    fetch_logic_gen u_fetch_logic (
        .clk                   (clk),
        .rst_n                 (rst_n),
        .start_fetch           (fetch_start),
        .reset_in_addr_counter (fetch_reset_addr), 
        .reset_wt_addr_counter (fetch_reset_addr),
        .Buffer_Select         (fetch_buffer_sel),
        .Tiles_Control         (fetch_tiles_ctrl),
        .Double_buffering      (fetch_double_buf),
        .hold_addr_ptr         (hold_addr_ptr),
        
        .bram_addr             (fetch_bram_addr),
        .bram_en               (fetch_bram_en),
        .fetch_done            (fetch_done)
    );

    // --- Write Logic ---
    write_logic_gen u_write_logic (
        .clk                (clk),
        .rst_n              (rst_n),
        .start_write        (write_start),
        .reset_addr_counter (write_reset_addr),
        .Buffer_Select      (write_buffer_sel),
        .Double_buffering   (write_double_buf),

        .bram_addr          (write_bram_addr),
        .bram_we            (write_bram_we),
        .write_done         (write_done),
        .write_tile_done    (write_tile_done),
        .busy               (write_busy)
    );

    // --- Q_K_V BRAM (Written ONLY by Hardware Write Logic) ---
    Q_K_V_Buffer u_qkv_buffer (
        // Port A (Write) - Controlled by write_logic_gen
        .clka   (clk),
        .ena    (write_bram_we),       
        .wea    (write_bram_we),       
        .addra  (write_bram_addr),
        .dina   (sa_data_out[255:0]), // Computed outputs from Systolic Array
        
        // Port B (Read) - Multiplexed between Testbench Read and Fetch Logic
        .clkb   (clk),
        .enb    (tb_read_mode ? tb_qkv_enb   : fetch_bram_en), 
        .addrb  (tb_read_mode ? tb_qkv_addrb : fetch_bram_addr), 
        .doutb  (doutb_qkv)       
    );

    // --- W_B_I BRAM (Written ONLY by Testbench Preloader) ---
    W_B_I_Buffer u_wbi_buffer (
        // Port A (Write) - Controlled by TB 
        .clka   (clk),
        .ena    (tb_wbi_we),
        .wea    (tb_wbi_we),
        .addra  (tb_wbi_addr),
        .dina   (tb_wbi_data),
        
        // Port B (Read) - Controlled by fetch_logic_gen
        .clkb   (clk),
        .enb    (fetch_bram_en),
        .addrb  (fetch_bram_addr),
        .doutb  (doutb_wbi)
    );

    // --- Systolic Array ---
    systolic_top u_systolic (
        .clk             (clk),
        .rst_n           (rst_n),
        .in_A            (doutb_qkv), // Depends on fetch phase; may need routing/mux if WBI holds initial inputs
        .weights         (doutb_wbi), 
        
        .valid_in        (reg_sa_valid_in),
        .load_weight     (reg_sa_load_weight),
        .first_iteration (reg_sa_first_iter),
        .last_tile       (reg_sa_last_tile),
        
        .ready           (sa_ready),
        .busy            (sa_busy),
        .done            (sa_done),
        .valid_out       (sa_valid_out),
        .data_out        (sa_data_out)
    );

    // =======+++++++++++++++++++++++++++++++++++++++++++++++++++==================================================================
    // Test Stimulus Sequence & Tasks
    // =======+++++++++++++++++++++++++++++++++++++++++++++++++++==================================================================
    integer i;
    integer row; // Used for initializing the matrices

    reg [255:0] matrix_32x32 [0:31]; 
    reg [255:0] matrix_512x32 [0:511];
    reg [255:0] output_matrix_512x32_Q [0:511]; // Matrix to store read-back Q/K/V tile
    reg [255:0] output_matrix_512x32_K [0:511];
    reg [255:0] output_matrix_512x32_V [0:511];

    // -------------------------------------------------------------------------
    // TASK: Write Weight (Loads 32x32 matrix into W_B_I Buffer)
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
                tb_wbi_data = data_matrix[w_idx]; // Read from the passed array
            end
            @(posedge clk);
            tb_wbi_we = 0; // Deassert write enable
        end
    endtask

    // -------------------------------------------------------------------------
    // TASK: Load Input (Loads 512x32 matrix into W_B_I Buffer)
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
                tb_wbi_data = data_matrix[in_idx]; // Read from the passed array
            end
            @(posedge clk);
            tb_wbi_we = 0; // Deassert write enable
        end
    endtask

    // -------------------------------------------------------------------------
    // TASK: Read Q_K_V Tile (Reads 512x32 matrix from Q_K_V Buffer)
    // -------------------------------------------------------------------------
    task automatic read_qkv_tile(
        input [15:0] start_addr,
        ref reg [255:0] read_matrix [0:511]
    );
        integer r_idx;
        begin
            $display("Task: Reading 512x32 Q_K_V Matrix from address %0d...", start_addr);
            
            // 1. Take control of the port
            tb_read_mode = 1'b1; 
            tb_qkv_enb   = 1'b1;
            
            // 2. Loop 513 times to account for the 1-cycle BRAM latency
            for (r_idx = 0; r_idx <= 512; r_idx = r_idx + 1) begin
                
                // Drive the address (for 0 to 511)
                if (r_idx < 512) begin
                    tb_qkv_addrb = start_addr + r_idx;
                end
                
                // Wait for the clock edge
                @(posedge clk);
                
                // Capture the data. Because of 1-cycle latency, data for address 'N' 
                // is available at loop iteration 'N+1'.
                if (r_idx > 0) begin
                    // Small delay to ensure data is perfectly stable after clock edge
                    #1; 
                    read_matrix[r_idx - 1] = doutb_qkv;
                end
            end
            
            // 3. Release control
            tb_qkv_enb   = 1'b0;
            tb_read_mode = 1'b0; 
            $display("Task: Read Complete.");
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

        // first weight input tile.
        for (row = 0; row < 32; row = row + 1) begin
           matrix_32x32[row] = {32{8'd1}};
        end
        for (row = 0; row < 512; row = row + 1) begin
            matrix_512x32[row] = {32{8'd2}};
        end
        write_weight(?,matrix_32x32);
        write_weight(?,matrix_32x32); // db
        load_input(?,matrix_512x32);  // db
        load_input(?,matrix_512x32);

        // the controller will start working here 


        wait(/* specific signal will be added in the controller for knowing the thae controller is done with writing Q,K,V 
            for depugining only*/);  
        
        dump_qkv_buffer_to_file("qkv_computed_results.txt");
        $finish;
    end

    // -------------------------------------------------------------------------
    // TASK: Dump Q_K_V Buffer to File (Reads the whole buffer to a text file)
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
                
                // Capture the data. Because of 1-cycle latency, data for address 'N' 
                // is available at loop iteration 'N+1'.
                if (r_idx > 0) begin
                    #1; // Small delay to ensure BRAM data is stable after clock edge
                    // Write the 256-bit data to the file in Hexadecimal format
                    $fdisplay(fd, "%h", doutb_qkv);
                end
            end
            
            // 3. Release control and close the file
            tb_qkv_enb   = 1'b0;
            tb_read_mode = 1'b0; 
            $fclose(fd);
            
            $display("Task: Dump Complete. File saved to %s", filename);
        end
    endtask

endmodule