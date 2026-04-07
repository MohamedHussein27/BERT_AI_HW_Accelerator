`timescale 1ns / 1ps

// signals to registerd by one cycle in the top module
// sa_valid_in,
// sa_load_weight,
// sa_first_iter,
// sa_last_tile,















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
    output logic       write_reset_address_counter
    input  logic       write_done,
    input  logic       write_tile_done,
    input  logic       write_busy,

    // ========================================================
    // 2. Systolic Array Interface (Shared for MHA & FFN)
    // ========================================================
    output logic       sa_valid_in,
    output logic       sa_load_weight,
    output logic       sa_first_iter,
    output logic       sa_last_tile,
    input  logic       sa_done,
    input  logic       sa_valid_out,

    // ========================================================
    // 3. Softmax Interface
    // ========================================================
    output logic       softmax_start,
    input  logic       softmax_done,

    // ========================================================
    // 4. LayerNorm Interface
    // ========================================================
    // Note: LayerNorm math is triggered by data flowing from fetch logic.
    // We just monitor it until the pipeline drains.
    input  logic       ln_valid_out,
    input  logic       ln_done,
    output logic       ln_valid_in
);

    // --------------------------------------------------------
    // FSM State Definitions: The Transformer Pipeline
    // --------------------------------------------------------
    typedef enum logic [3:0] {
        ST_IDLE          = 4'd0,
        
        // --- Multi-Head Attention (MHA) ---
        FETCHING_W      = 4'd1, // Q, K, V Generation (Systolic Array)
        ST_MHA_SCORE     = 4'd2, // Q x K^T (Systolic Array)
        ST_MHA_SOFTMAX   = 4'd3, // Softmax Streaming
        ST_MHA_CONTEXT   = 4'd4, // Score x V (Systolic Array)
        ST_MHA_ADD_NORM  = 4'd5, // Residual Add + LayerNorm
        
        // --- Feed-Forward Network (FFN) ---
        ST_FFN_LINEAR1   = 4'd6, // FFN Expansion (Systolic Array)
        ST_FFN_GELU      = 4'd7, // GELU Activation (Streaming)
        ST_FFN_LINEAR2   = 4'd8, // FFN Projection (Systolic Array)
        ST_FFN_ADD_NORM  = 4'd9, // Residual Add + LayerNorm
        
        ST_DONE          = 4'd10
    } master_state_e;

    master_state_e state, next_state;
    
    // counters
    logic [4:0] sa_first_iter_counter; 
    logic [4:0] done_counter;
    logic [1:0] Q_K_V_sel;  // this signal tells whethere we are in Q or K or V
    // 0 --> Q, 1 --> K, 2 --> 
    logic [4:0] done_in_tile_counter; // to count how many input tiles we fetched
    logic [4:0] done_wt_tile_counter; // to count how many weight tiles we fetched
    logic [5:0] piso_counter;         // counter to wait until the serialized inputs go to the softmax to fetch a new 32 elements

    // flags
    logic       Q_Kt_sel; // this signal like the above makes us write in the Q_Kt buffer
    logic       repeat_matrix; // this signal is for the cpu to fetch the matrix tiles again for sa
    logic db_control;  // to control when to fetch from the double buffering addresses
    //logic first_sa_time; // flag to raise the first iteration output for the sa in the first time only

    logic write_pulse_flag; // flag to make the write to be high only for one cycle



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

            // falgs
            write_pulse_flag <= 1;
        end
        else begin        
            state <= next_state;
        end

        if ((state == FETCHING_I || state == FETCHING_K_I || state == FETCHING_V_I) 
            && 
            (next_state == FETCHING_W || next_state == FETCHING_K_W || next_state == FETCHING_V_W)) db_control <= 1;
        else                                                                                       db_control <= 0;

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
                fetch_reset_in_addr_counter <= 1'
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

        // inceremnt piso counter to go back to fetching Q_Kt state
        if (state == WAIT_PISO) 
            piso_counter <= piso_counter + 1; // can be handeled by busy signal from piso module
        else
            piso_counter <= '0;
            

        

        //**************************************** flags ************************************\\
        if (sa_valid_out && write_pulse_flag) begin
            write_start <= 1;
            write_pulse_flag <= 0;
        end
        else begin
            write_start <= 0;
        end

    end

    // ========================================================
    // Block 2: Combinational Next-State Logic (The Handshakes)
    // ========================================================
    always_comb begin
        next_state = state; // Default hold
        
        case (state)
            ST_IDLE:        if (start_inference) next_state = FETCHING_W;
            
            // First Stage: filling Q K V matrices to be used 
            FETCHING_W:     if (fetch_done)             next_state = FETCHING_I;
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
                            if (write_done && Q_K_V_sel == 3) next_state = FETCHING_Kt;
                            else if (write_done)              next_state = FETCHING_W;
                        end


            // second stage: filling Q Kt matrix to be used in softmax
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
                            if (write_done && Q_Kt_sel) next_state = FETCHING_Q_Kt;
                            else if (write_done)              next_state = FETCHING_Kt;
                        end

            // third stage: softmax
            FETCHING_Q_Kt: if (fetch_in_done) next_state = WAIT_SIPO;
                           else next_state = WAIT_PISO; // wait the 32 clocks as the 32 elements being serialized to the softmax
            
            WAIT_PISO: 
                        begin
                            if (piso_counter == 16) next_state = FETCHING_Q_Kt;
                            else                    next_state = WAIT_PISO;
                        end

            WAIT_SIPO:

            FETCHING_SM:

            FETCHING_V:

            WRITING_SV:


            // fourth stage: writing the multi-head attention matrix
            FETCHING_SV:

            //FETCHING_W:

            WRITING_H:

            // fifth stage: layernormalization
            RES_ADD_H_I:

            WRITING_LN:

            FETCHING_LN:

            FETCHING_LN_W:

            WRITING_FFN_I:


            ST_MHA_SCORE:    if (fetch_done)         next_state = ST_MHA_SOFTMAX;
            
            // Wait for Softmax to finish streaming
            ST_MHA_SOFTMAX:  if (softmax_done)    next_state = ST_MHA_CONTEXT;
            
            ST_MHA_CONTEXT:  if (fetch_done)         next_state = ST_MHA_ADD_NORM;
            
            // Wait for LayerNorm to finish its 3-pass process
            ST_MHA_ADD_NORM: if (ln_done)         next_state = ST_FFN_LINEAR1;
            
            ST_FFN_LINEAR1:  if (fetch_done)         next_state = ST_FFN_GELU;
            
            // GELU is purely combinational , so we just wait for memory write to finish
            ST_FFN_GELU:     if (write_done)      next_state = ST_FFN_LINEAR2;
            
            ST_FFN_LINEAR2:  if (fetch_done)         next_state = ST_FFN_ADD_NORM;
            ST_FFN_ADD_NORM: if (ln_done)         next_state = ST_DONE;
            
            ST_DONE:                              next_state = ST_IDLE;
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
        

        // write logic
        write_start      = 1'b0;

        // Default SA Control
        sa_valid_in      = 1'b0;
        sa_load_weight   = 1'b0;
        sa_first_iter    = 1'b0;
        sa_last_tile     = 1'b0;
        //first_sa_time    = 1'b1;


        softmax_start    = 1'b0;

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
                    sa_first_iter = 1'b1;  // we need it high only for the first iteration (as we dont have any partial sums)
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
                    sa_first_iter = 1'b1;  // we need it high only for the first iteration (as we dont have any partial sums)
                end
                sa_valid_in = 1'b1;
            end



            ST_MHA_SOFTMAX: begin
                // Fetch Scores, stream directly to bert_softmax.sv [cite: 131-133]
                fetch_buffer_sel = 4'b0110; 
                softmax_start    = 1'b1;    // Kick off softmax FSM [cite: 158]
            end

            ST_MHA_CONTEXT: begin
                // Fetch Softmax output and V [cite: 51]
                fetch_buffer_sel = 4'b0111; 
                sa_valid_in      = 1'b1;
            end

            ST_MHA_ADD_NORM: begin
                // Fetch from Attention output buffer and route to LayerNorm
                // Tile control set to 24 for 3-pass routing [cite: 61]
                fetch_buffer_sel = 4'b1000; 
                fetch_tiles_ctrl = 2'b10;   
                fetch_start      = 1'b1;
            end

            // ----------------------------------------------------
            // FEED FORWARD NETWORK
            // ----------------------------------------------------
            ST_FFN_LINEAR1: begin
                // Fetch FFN Input [cite: 58] and FFN Weights [cite: 56]
                fetch_buffer_sel = 4'b1011; 
                sa_valid_in      = 1'b1;
            end

            ST_FFN_GELU: begin
                // Fetch Intermediate FFN Output [cite: 59]
                // Streams through purely combinational GCU.sv 
                fetch_buffer_sel = 4'b1100;
                fetch_start      = 1'b1;
                write_start      = 1'b1; // Trigger write_logic to catch GELU output
            end

            ST_FFN_LINEAR2: begin
                // Fetch GELU outputs and FFN2 Weights
                fetch_buffer_sel = 4'b1100; // Adjust mapping to point to GELU out
                sa_valid_in      = 1'b1;
            end

            ST_FFN_ADD_NORM: begin
                // Stream final FFN projection to LayerNorm
                fetch_buffer_sel = 4'b1101; // Output buffer [cite: 60]
                fetch_tiles_ctrl = 2'b10;   // 24 chunks [cite: 61]
                fetch_start      = 1'b1;
            end

            ST_DONE: begin
                layer_done = 1'b1;
            end
        endcase
    end
endmodule