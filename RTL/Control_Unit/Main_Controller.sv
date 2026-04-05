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
    input  logic       fetch_done,

    output logic [3:0] write_buffer_sel,
    output logic       write_start,
    input  logic       write_done,

    // ========================================================
    // 2. Systolic Array Interface (Shared for MHA & FFN)
    // ========================================================
    output logic       sa_valid_in,
    output logic       sa_load_weight,
    output logic       sa_first_iter,
    output logic       sa_last_tile,
    input  logic       sa_done,

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
    logic db_control;  // to control when to fetch from the double buffering addresses
    //logic first_sa_time; // flag to raise the first iteration output for the sa in the first time only

    logic [4:0] sa_first_iter_counter; 
    logic [4:0] done_counter;
    logic [1:0] Q_K_V_sel;  // this signal tells whethere we are in Q or K or V
    // 0 --> Q, 1 --> K, 2 --> V


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
        end
        else begin        
            state <= next_state;
        end

        if ((state == FETCHING_I || state == FILLING_K_I || state == FILLING_V_I) 
            && 
            (next_state == FETCHING_W || next_state == FILLING_K_W || next_state == FILLING_V_W)) db_control <= 1;
        else                                                                                       db_control <= 0;

        //*********************************** counters *********************************\\
        // this counter is used to count when the code is back to the first tile in the input 
        // to generate first_tile signal for SA.
        if (((next_state == FETCHING_W && state == FETCHING_I) || (next_state == FILLING_K_W && state == FILLING_K_I)
            || (next_state == FILLING_V_W && state == FILLING_V_I))) begin
            if (sa_first_iter_counter <= 23)
                sa_first_iter_counter <= sa_first_iter_counter + 1;
            else sa_first_iter_counter <= '0;
        end
        // this counter is used to count the number of dones 
        // if its 24 then we know that a matrix is done and we should change direction to calculate the next matrix
        if (sa_done) begin 
            done_counter <= done_counter + 1;
        end

        if (done_counter == 24)begin
            done_counter <= 0;
            Q_K_V_sel <= Q_K_V_sel + 1;
        end

    end

    // ========================================================
    // Block 2: Combinational Next-State Logic (The Handshakes)
    // ========================================================
    always_comb begin
        next_state = state; // Default hold
        
        case (state)
            ST_IDLE:         if (start_inference) next_state = FETCHING_W;
            
            // start filling buffers
            FETCHING_W:     if (fetch_done)             next_state = FETCHING_I;
            FETCHING_I:     if (fetch_done && !sa_done)    next_state = FILLING_W;
                            else                        next_state = WRITING_Q_K_V;
            WRITING_Q_K_V:  if (write_done && Q_K_V_sel == 3) next_state = // QKT
                            else if (write_done)              next_state = FETCHING_W;

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
        
        // Default Memory Control
        fetch_start      = 1'b0;
        fetch_buffer_sel = 4'b0000;
        fetch_tiles_ctrl = 2'b00;
        fetch_double_buf = 1'b0;
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
                // Example: Fetch Weights 'W' 
                fetch_buffer_sel = 4'b0000;  
                fetch_tiles_ctrl = 2'b01;  // fetch 32
                fetch_double_buf = db_control;
                
                // Wake up Systolic Array
                sa_load_weight = 1'b1;
            end

            FETCHING_I: begin
                // Example: Fetch Weights 'I'
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
                    write_start = 1;
                end
                else if (Q_K_V_sel == 1) begin
                    write_buffer_sel = 4'b0100;
                    write_start = 1;
                end
                else if (Q_K_V_sel == 2) begin
                    write_buffer_sel = 4'b0101;
                    write_start = 1;
                end
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