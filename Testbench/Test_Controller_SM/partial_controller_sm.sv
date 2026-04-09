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
    output logic       write_sipo_mode, // for making the write logic inceremnt address only when sipo_valid_out
    input  logic       write_done_all,
    input  logic       write_tile_done,
    input  logic       write_busy,

    // ========================================================
    // 2. Systolic Array Interface (Missing ports added for logic)
    // ========================================================
    input  logic       sa_done,
    input  logic       sa_valid_out,
    output logic       sa_valid_in,
    output logic       sa_load_weight,
    output logic       sa_first_iter,
    output logic       sa_last_tile,

    // ========================================================
    // 3. Softmax Interface
    // ========================================================
    input  logic       piso_valid_out,
    input  logic       piso_busy,
    output logic       softmax_start,
    input  logic       softmax_done,
    input  logic       softmax_out_valid, // Added: Used in WAIT_SOFTMAX
    input  logic       softmax_out_last,
    output logic       softmax_out_in,    // Added: Driven in WAIT_SIPO

    // ========================================================
    // 4. Quantization
    // ========================================================
    // Vector Quantizaiton
    output logic       quantize_valid_in,
    output logic       quantize_param_addr
    // Element Quantization
    output logic       quantize_u_valid_in
);

    // --------------------------------------------------------
    // FSM State Definitions: The Transformer Pipeline
    // Expanded to 5 bits to fit all states
    // --------------------------------------------------------
    typedef enum logic [4:0] {
        ST_IDLE          = 5'd0,
        
        // --- Multi-Head Attention (MHA) ---
        FETCHING_W       = 5'd1, 
        FETCHING_I       = 5'd2,
        WRITING_Q_K_V    = 5'd3,
        FETCHING_Kt      = 5'd4,
        FETCHING_Q       = 5'd5,
        WRITING_Q_Kt     = 5'd6,
        FETCHING_Q_Kt    = 5'd7,
        WAIT_PISO        = 5'd8,
        WAIT_SOFTMAX     = 5'd9,
        WAIT_SIPO        = 5'd10,
        FETCHING_V       = 5'd11,
        
        // --- Placeholders for future states mentioned in comments ---
        FETCHING_K_I     = 5'd12,
        FETCHING_V_I     = 5'd13,
        FETCHING_K_W     = 5'd14,
        FETCHING_V_W     = 5'd15,

        // --- Original States ---
        ST_MHA_SCORE     = 5'd16, 
        ST_MHA_SOFTMAX   = 5'd17, 
        ST_MHA_CONTEXT   = 5'd18, 
        ST_MHA_ADD_NORM  = 5'd19, 
        
        // --- Feed-Forward Network (FFN) ---
        ST_FFN_LINEAR1   = 5'd20, 
        ST_FFN_GELU      = 5'd21, 
        ST_FFN_LINEAR2   = 5'd22, 
        ST_FFN_ADD_NORM  = 5'd23, 
        
        ST_DONE          = 5'd24
    } master_state_e;

    master_state_e state, next_state;
    
    // counters
    logic [4:0] sa_first_iter_counter; 
    logic [4:0] done_counter;
    logic [1:0] Q_K_V_sel;  // this signal tells whethere we are in Q or K or V (0 -> Q, 1 -> K, 2 -> V)
    logic [4:0] done_in_tile_counter; // to count how many input tiles we fetched
    logic [4:0] done_wt_tile_counter; // to count how many weight tiles we fetched
    logic [5:0] piso_counter;         // counter to wait until the serialized inputs go to the softmax to fetch a new 32 elements

    // flags
    logic       Q_Kt_sel; // this signal like the above makes us write in the Q_Kt buffer
    logic       repeat_matrix; // this signal is for the cpu to fetch the matrix tiles again for sa
    logic       db_control;  // to control when to fetch from the double buffering addresses
    //logic     first_sa_time; // flag to raise the first iteration output for the sa in the first time only

    logic write_pulse_flag; // flag to make the write to be high only for one cycle
    logic softmax_pulse_flag; // flag to make softmax start sig high for one clock


    //

    assign quantize_u_valid_in = softmax_out_valid;

    //
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
            Q_Kt_sel <= '0;
            done_in_tile_counter <= '0;
            done_wt_tile_counter <= '0;

            // Sequential Outputs Reset
            sa_last_tile <= 0;
            write_start <= 0;
            softmax_start <= 0;
            fetch_reset_in_addr_counter <= 0;
            fetch_reset_wt_addr_counter <= 0;

            // flags
            write_pulse_flag <= 1;
            softmax_pulse_flag <= 1;
        end
        else begin        
            state <= next_state;
            
            // Pulse defaults (prevents these from latching high forever)
            fetch_reset_in_addr_counter <= 1'b0;
            write_start <= 1'b0;
            softmax_start <= 1'b0;

            if ((state == FETCHING_I || state == FETCHING_K_I || state == FETCHING_V_I) 
                && 
                (next_state == FETCHING_W || next_state == FETCHING_K_W || next_state == FETCHING_V_W)) db_control <= 1;
            else                                                                                         db_control <= 0;

            //*********************************** counters *********************************\\
            // this counter is used to count when the code is back to the first tile in the input 
            // to generate first_tile signal for SA.
            if (((next_state == FETCHING_W && state == FETCHING_I) || (next_state == FETCHING_Kt && state == FETCHING_Q)
                /*|| (next_state == FETCHING_V_W && state == FETCHING_V_I)*/)) begin
                if (sa_first_iter_counter <= 23)
                    sa_first_iter_counter <= sa_first_iter_counter + 1;
                else sa_first_iter_counter <= '0;
            end
            
            // this counter is used to count the number of dones 
            // if its 24 then we know that a matrix is done and we should change direction to calculate the next matrix
            if (sa_done) begin 
                done_counter <= done_counter + 1;
            end

            // condition for dividing the heads and write the Q_Kt buffer 12 times (512 x 64) x (64 x 512)
            if (done_counter == 2 && (state == FETCHING_Kt || state == FETCHING_Q || state == WRITING_Q_Kt)) 
                sa_last_tile <= 1;

            // to make the sa outputs its valid outputs
            else if (done_counter == 23) sa_last_tile <= 1; 

            // transition to write if we are in the last tile
            else if (done_counter == 24)begin
                done_counter <= 0;
                if (state == FETCHING_W || state == FETCHING_I || state == WRITING_Q_K_V) begin
                    if (Q_K_V_sel == 3) Q_K_V_sel <= 0;
                    else                Q_K_V_sel <= Q_K_V_sel + 1;
                end
                else if (state == FETCHING_Kt || state == FETCHING_Q || state == WRITING_Q_Kt) begin
                    /*if (Q_Kt_sel == 1) Q_Kt_sel <= 0;
                    else                Q_Kt_sel <= Q_Kt_sel + 1;*/
                    Q_Kt_sel <= ~Q_Kt_sel;
                end
                sa_last_tile <= 0;
            end

            // multiplying (512 x 64) x (64 x 512) conditions
            if (fetch_in_done) begin
                if (done_in_tile_counter == 23)
                    done_in_tile_counter <= '0;
                // condition for multiplying 512 x 64 (inputs) so we need two tiles and then fetch them again
                else if ((state == FETCHING_Kt || state == FETCHING_Q || state == WRITING_Q_Kt) && (done_in_tile_counter == 2)) begin
                    done_in_tile_counter <= '0;
                    fetch_reset_in_addr_counter <= 1'b1; // FIXED: syntax error 1'
                end
                else
                    done_in_tile_counter <= done_in_tile_counter + 1;
            end
            
            if (fetch_wt_done) begin
                if (done_wt_tile_counter == 23)
                    done_wt_tile_counter <= '0;
                else
                    done_wt_tile_counter <= done_wt_tile_counter + 1;
            end

            // incerement piso counter to go back to fetching Q_Kt state
            /*if (state == WAIT_PISO) 
                piso_counter <= piso_counter + 1; // can be handeled by busy signal from piso module
            else
                piso_counter <= '0;*/
            

            //**************************************** flags ************************************\\
            // pulsing write start
            if (sa_valid_out && write_pulse_flag) begin
                write_start <= 1;
                write_pulse_flag <= 0;
            end
            else if (next_state == WRITING_Q_K_V || next_state == WRITING_Q_Kt) begin
                write_pulse_flag <= 1;
            end
            else begin
                write_start <= 0;
            end

            // pulsing softmax start
            if (piso_valid_out && softmax_pulse_flag) begin
                softmax_start <= 1;
                softmax_pulse_flag <= 0;
            end
            else if (state == FETCHING_Q_Kt) begin
                softmax_pulse_flag <= 1;
            end
            else begin
                softmax_start <= 0;
            end

            /*if (state == WRITING_Q_Kt)
                fetch_stop_counting <= 1;
            else 
                fetch_stop_counting <= 0;*/

        end
    end

    // ========================================================
    // Block 2: Combinational Next-State Logic (The Handshakes)
    // ========================================================
    always_comb begin
        next_state = state; // Default hold
        
        case (state)
            ST_IDLE:        if (start_inference) next_state = FETCHING_W;
            
            //******************** First Stage: filling Q K V matrices to be used ***********************\\
            FETCHING_W:     if (fetch_done)             next_state = FETCHING_Q_Kt;
            FETCHING_I: 
                        begin   
                            if (sa_done) begin
                                if (!sa_last_tile)  
                                    next_state = FETCHING_W;
                                else   
                                    next_state = WRITING_Q_K_V;
                            end
                        end 
            WRITING_Q_K_V: 
                        begin  
                            if (write_done_all && Q_K_V_sel == 3) next_state = FETCHING_Kt;
                            else if (write_done_all)              next_state = FETCHING_W;
                        end


            //******************** second stage: filling Q Kt matrix to be used in softmax ******************\\
            FETCHING_Kt:    if (fetch_done)             next_state = FETCHING_Q; // Kt is treated as weights
            FETCHING_Q: begin   
                            if (sa_done) begin
                                if (!sa_last_tile)  
                                    next_state = FETCHING_Kt;
                                else   
                                    next_state = WRITING_Q_Kt;
                            end
                        end
            
            WRITING_Q_Kt:
                        begin  
                            if (write_done_all && Q_Kt_sel) next_state = FETCHING_Q_Kt;
                            else if (write_done_all)              next_state = FETCHING_Kt;
                        end

            //********************* third stage: softmax ****************************\\
            FETCHING_Q_Kt: if (fetch_in_done) next_state = WAIT_SOFTMAX; // wait softmax processing the row and then give it another
                           else if (fetch_stop_counting) next_state = WAIT_PISO;
                           else next_state = FETCHING_Q_Kt; // wait the 32 clocks as the 32 elements being serialized to the softmax
            
            WAIT_PISO: 
                        begin
                            /*if (piso_counter == 16) next_state = FETCHING_Q_Kt;
                            else                    next_state = WAIT_PISO;*/
                            if (!piso_busy) next_state = FETCHING_Q_Kt;
                            else            next_state = WAIT_PISO;
                        end
            // we are processing the complete row in softmax
            WAIT_SOFTMAX: if (softmax_out_valid) next_state = WAIT_SIPO;
                          else next_state = WAIT_SOFTMAX;
            
            // assuming the complete row is being outputed serially
            WAIT_SIPO: 
                        begin    
                            if (write_done_all) next_state = FETCHING_V;
                            else if (softmax_out_last) next_state = FETCHING_Q_Kt;
                            else next_state = WAIT_SIPO;
                        end
            
            default:                              next_state = ST_IDLE;
        endcase
    end

    // ========================================================
    // Block 3: Combinational Output Mapping
    // ========================================================
    always_comb begin
        // Default System Hooks
        layer_done       = 1'b0;
        
        // Fetch logic
        fetch_start      = 1'b0;
        fetch_buffer_sel = 4'b0000;
        fetch_tiles_ctrl = 2'b00;
        fetch_double_buf = 1'b0;
        fetch_hold_addr_ptr = 1'b0; // Added default
        fetch_stop_counting = 1'b0;

        // write logic
        write_buffer_sel = 4'd0;    // Added default
        write_double_buf = 1'b0;    // Added default
        write_sipo_mode  = 1'b0;    // Added default
        write_reset_address_counter = 1'b0; // Added default

        // Default SA Control
        sa_valid_in      = 1'b0;
        sa_load_weight   = 1'b0;
        sa_first_iter    = 1'b0;

        // quantization
        quantize_valid_in = 1'b0;
        quantize_param_addr = 1'b0; // Added default

        // softmax serializers
        softmax_out_in = 1'b0;

        // Note: write_start, softmax_start, and sa_last_tile are driven sequentially 
        // in Block 1, so they are explicitly NOT defaulted here to prevent multiple drivers.

        case (state)
            // ----------------------------------------------------
            // MULTI-HEAD ATTENTION
            // ----------------------------------------------------
            FETCHING_W: begin
                // Fetch Weights 'W' 
                fetch_buffer_sel = 4'b0000;  
                fetch_tiles_ctrl = 2'b01;  // fetch 32
                fetch_double_buf = db_control;
                
                // Wake up Systolic Array
                sa_load_weight = 1'b1;
            end

            FETCHING_I: begin
                //  Fetch Weights 'I'
                fetch_buffer_sel = 4'b0010; 
                fetch_tiles_ctrl = 2'b00; // fetch 512
                fetch_double_buf = db_control;
                
                // Wake up Systolic Array
                if (sa_first_iter_counter == 0) begin
                    sa_first_iter = 1'b1;  // we need it high only for the first iteration
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

            FETCHING_Kt: begin
                // Fetch Weights 'W' 
                fetch_buffer_sel = 4'b0100;  
                fetch_tiles_ctrl = 2'b01;  // fetch 32
                
                // Wake up Systolic Array
                sa_load_weight = 1'b1;
            end

            FETCHING_Q: begin
                //  Fetch Weights 'I'
                fetch_buffer_sel = 4'b0011; 
                fetch_tiles_ctrl = 2'b00; // fetch 512
                
                // Wake up Systolic Array
                if (sa_first_iter_counter == 0) begin
                    sa_first_iter = 1'b1;  
                end
                sa_valid_in = 1'b1;
            end
            
            FETCHING_Q_Kt: begin // we dont need sa here
                //  Fetch Weights 'Softmax'
                fetch_buffer_sel = 4'b0110; 
                fetch_tiles_ctrl = 2'b11; // fetch 16
            end
                
            WAIT_PISO: begin
                quantize_valid_in = 1'b1;
                fetch_stop_counting = 1'b1;
                quantize_param_addr = 8'h04; // addr to softmax scales
            end
            
            WAIT_SOFTMAX: begin
                // holding state 
            end

            WAIT_SIPO: begin
                softmax_out_in = 1'b1;
                write_sipo_mode = 1'b1;
            end
            
            default: ; // Defaults handle everything else safely
        endcase
    end
endmodule