`timescale 1ns / 1ps

module transformer_master_ctrl (
    input  logic clk,
    input  logic rst_n,
    
    // Top-Level Execution
    input  logic start_inference,
    output logic layer_done,

    // ========================================================
    // 1. Fetch & Write Logic Interface
    // ========================================================
    output logic       fetch_start,
    output logic [3:0] fetch_buffer_sel, // Determines which SRAM to read
    output logic [1:0] fetch_tiles_ctrl, // 512, 32, or 24 tile configuration
    output logic       fetch_double_buf,
    output logic       fetch_hold_addr_ptr,  // for layernorm to fetch the same address 
    output logic       fetch_reset_in_addr_counter,
    output logic       fetch_reset_wt_addr_counter,
    output logic       fetch_stop_counting,
    input  logic       fetch_in_done,
    input  logic       fetch_wt_done,
    input  logic       fetch_busy,

    output logic [3:0] write_buffer_sel,
    output logic       write_start,
    output logic       write_double_buf,
    output logic       write_reset_address_counter,
    input  logic       write_done_all,
    input  logic       write_tile_done,
    input  logic       write_busy,

    // ========================================================
    // 2. Systolic Array Interface
    // ========================================================
    output logic       sa_valid_in,
    output logic       sa_load_weight,
    output logic       sa_first_iter,
    output logic       sa_last_tile,
    input  logic       sa_done,
    input  logic       sa_valid_out
);

    // --------------------------------------------------------
    // FSM State Definitions: Stage 1 (Q, K, V Generation) Only
    // --------------------------------------------------------
    typedef enum logic [1:0] {
        ST_IDLE       = 2'd0,
        FETCHING_W    = 2'd1,
        FETCHING_I    = 2'd2,
        WRITING_Q_K_V = 2'd3
    } master_state_e;

    master_state_e state, next_state;
    
    // counters
    logic [4:0] sa_first_iter_counter; 
    logic [4:0] done_counter;
    logic [1:0] Q_K_V_sel;  // 0 --> Q, 1 --> K, 2 --> V
    logic [4:0] done_in_tile_counter; // to count how many input tiles we fetched
    logic [4:0] done_wt_tile_counter; // to count how many weight tiles we fetched

    // flags
    logic repeat_matrix; // this signal is for the cpu to fetch the matrix tiles again for sa
    logic db_control;    // to control when to fetch from the double buffering addresses
    logic write_pulse_flag; // flag to make the write to be high only for one cycle
    logic fetch_pulse_flag; // flag to make the fetch to be high only for one cycle

    // ========================================================
    // Block 1: Sequential State Memory
    // ========================================================
    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin 
            state <= ST_IDLE;
            db_control <= 0;
            sa_first_iter_counter <= '0;
            done_counter <= '0;
            Q_K_V_sel <= '0;
            done_in_tile_counter <= '0;
            done_wt_tile_counter <= '0;
            fetch_reset_in_addr_counter <= 0;
            sa_last_tile <= 0;

            // flags
            write_pulse_flag <= 1;
            fetch_pulse_flag <= 1;
            fetch_stop_counting <= 0;
        end
        else begin        
            state <= next_state;

            // db_control logic 
            if (state == FETCHING_I && next_state == FETCHING_W) 
                db_control <= 1;
            else                                                                                       
                db_control <= 0;

            //*********************************** counters *********************************\\
            // sa_first_iter_counter
            if (next_state == FETCHING_W && state == FETCHING_I) begin
                if (sa_first_iter_counter <= 23)
                    sa_first_iter_counter <= sa_first_iter_counter + 1;
                else 
                    sa_first_iter_counter <= '0;
            end

            // done_counter
            if (sa_done) begin 
                done_counter <= done_counter + 1;
            end

            // to make the sa outputs its valid outputs
            if (done_counter == 23) 
                sa_last_tile <= 1; 

            // transition to write if we are in the last tile
            else if (done_counter == 24) begin
                done_counter <= 0;
                if (state == FETCHING_W || state == FETCHING_I || state == WRITING_Q_K_V) begin
                    if (Q_K_V_sel == 3) Q_K_V_sel <= 0;
                    else                Q_K_V_sel <= Q_K_V_sel + 1;
                end
                sa_last_tile <= 0;
            end

            // input tile counters
            if (fetch_in_done) begin
                if (done_in_tile_counter == 23)
                    done_in_tile_counter <= '0;
                else
                    done_in_tile_counter <= done_in_tile_counter + 1;
            end

            // weight tile counters
            if (fetch_wt_done) begin
                if (done_wt_tile_counter == 23)
                    done_wt_tile_counter <= '0;
                else
                    done_wt_tile_counter <= done_wt_tile_counter + 1;
            end

            //**************************************** flags ************************************\\
            // pulsing write start
            if (sa_valid_out && write_pulse_flag) begin
                write_start <= 1;
                write_pulse_flag <= 0;
            end
            else if (next_state == WRITING_Q_K_V) begin
                write_pulse_flag <= 1;
            end
            else begin
                write_start <= 0;
            end



            if ( fetch_pulse_flag && (next_state == FETCHING_W || next_state == FETCHING_I)) begin
                fetch_start <= 1;
                fetch_pulse_flag <= 0;
            end
            else if (sa_done || fetch_wt_done ) begin
                fetch_pulse_flag <= 1;
            end
            else begin
                fetch_start <= 0;
            end
        end
    end

    // ========================================================
    // Block 2: Combinational Next-State Logic
    // ========================================================
    always_comb begin
        next_state = state; // Default hold
        
        case (state)
            ST_IDLE:        
                if (start_inference) next_state = FETCHING_W;
            
            //******************** First Stage: filling Q K V matrices ***********************\\
            FETCHING_W:     
                if (fetch_wt_done) next_state = FETCHING_I;
            
            FETCHING_I: begin   
                if (sa_done) begin
                    if (!sa_last_tile)  
                        next_state = FETCHING_W;
                    else   
                        next_state = WRITING_Q_K_V;
                end
            end 
            
            WRITING_Q_K_V: begin  
                if (write_done_all && Q_K_V_sel == 3) next_state = ST_IDLE;
                else if (write_done_all)              next_state = FETCHING_W;
            end
            
            default: next_state = ST_IDLE;
        endcase
    end

    // ========================================================
    // Block 3: Combinational Output Mapping
    // ========================================================
    always_comb begin
        // Default System Hooks
        layer_done       = 1'b0;
        
        // Fetch logic defaults
        //fetch_start      = 1'b0;
        fetch_buffer_sel = 4'b0000;
        fetch_tiles_ctrl = 2'b00;
        fetch_double_buf = 1'b0;
        fetch_hold_addr_ptr = 1'b0;
        fetch_reset_wt_addr_counter = 1'b0;
        
        // Write logic defaults
        //write_start      = 1'b0;
        write_buffer_sel = 4'b0000;
        write_double_buf = 1'b0;
        write_reset_address_counter = 1'b0;

        // Default SA Control
        sa_valid_in      = 1'b0;
        sa_load_weight   = 1'b0;
        sa_first_iter    = 1'b0;


        case (state)
            FETCHING_W: begin
                // Fetch Weights 'W' 
                fetch_buffer_sel = 4'b0000;  
                fetch_tiles_ctrl = 2'b01;  // fetch 32
                fetch_double_buf = db_control;
                
                // Wake up Systolic Array
                if (next_state == FETCHING_I) sa_load_weight = 1'b0;
                else sa_load_weight = 1'b1;
            end

            FETCHING_I: begin
                // Fetch Inputs 'I'
                fetch_buffer_sel = 4'b0010; 
                fetch_tiles_ctrl = 2'b00; // fetch 512
                fetch_double_buf = db_control;
                
                // Wake up Systolic Array
                if (sa_first_iter_counter == 0) begin
                    sa_first_iter = 1'b1;  // high only for the first iteration (no partial sums)
                end
                sa_valid_in = 1'b1;
            end

            WRITING_Q_K_V: begin
                if (Q_K_V_sel == 0) begin
                    write_buffer_sel = 4'b0011;
                end
                else if (Q_K_V_sel == 1) begin
                    write_buffer_sel = 4'b0100;
                end
                else if (Q_K_V_sel == 2) begin
                    write_buffer_sel = 4'b0101;
                end
            end
        endcase
    end
endmodule