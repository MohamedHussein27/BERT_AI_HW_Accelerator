`timescale 1ns / 1ps
import PE_pkg::*;

module layernorm_fsm (
    input  logic       clk,
    input  logic       rst_n,
    
    // Handshake from Main System Controller
    input  logic       data_valid, 

    // Internal PE & Datapath Control
    output pe_op_e     pe_opcode,
    output logic       accum_en,
    output logic       accum_fetch,
    
    // SQRT Control
    output logic       sqrt_valid_in,
    input  logic       sqrt_valid_out,
    output logic       sqrt_busy, // for indicating square root module is still calculating
    
    // Top-Level Status
    output logic       out_valid,
    output logic       done
);

    typedef enum logic [2:0] {
        ST_PASS1_MEAN,
        ST_LOAD_MEAN,
        ST_PASS2_VAR,
        ST_TRIG_SQRT,
        ST_CALC_SQRT,
        ST_PASS3_NORM
    } state_t;

    state_t state, next_state;

    logic [4:0] chunk_cnt; // 0 to 23
    logic [8:0] row_cnt;   // 0 to 511

    // ========================================================
    // Block 1: Sequential Logic (State Memory & Counters)
    // ========================================================
    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            state     <= ST_PASS1_MEAN;
            chunk_cnt <= '0;
            row_cnt   <= '0;
        end else begin
            state <= next_state;
            
            // Only increment chunk_cnt when valid data is actively being processed
            if (data_valid && (state == ST_PASS1_MEAN || state == ST_PASS2_VAR || state == ST_PASS3_NORM)) begin
                if (chunk_cnt == 5'd23) 
                    chunk_cnt <= '0;
                else                    
                    chunk_cnt <= chunk_cnt + 1;
            end

            // Increment row counter at the very end of Pass 3
            if (state == ST_PASS3_NORM && data_valid && chunk_cnt == 5'd23) begin
                row_cnt <= row_cnt + 1;
            end
        end
    end

    // ========================================================
    // Block 2: Combinational Logic (Next State Routing)
    // ========================================================
    always_comb begin
        // Default assignment to prevent latches
        next_state = state; 
        
        case (state)
            ST_PASS1_MEAN: if (data_valid && chunk_cnt == 5'd23) next_state = ST_LOAD_MEAN;
            
            ST_LOAD_MEAN:  next_state = ST_PASS2_VAR; // 1-cycle automatic transition
            
            ST_PASS2_VAR:  if (data_valid && chunk_cnt == 5'd23) next_state = ST_TRIG_SQRT;
            
            ST_TRIG_SQRT:  next_state = ST_CALC_SQRT; // 1-cycle automatic transition
            
            ST_CALC_SQRT:  if (sqrt_valid_out) next_state = ST_PASS3_NORM;
            
            ST_PASS3_NORM: if (data_valid && chunk_cnt == 5'd23) next_state = ST_PASS1_MEAN;
            
            default:       next_state = ST_PASS1_MEAN;
        endcase
    end

    // ========================================================
    // Block 3: Combinational Logic (Outputs)
    // ========================================================
    always_comb begin
        // Default assignments to prevent latches
        pe_opcode     = OP_PASS_X;
        accum_en      = 1'b0;
        accum_fetch   = 1'b0;
        sqrt_valid_in = 1'b0;
        out_valid     = 1'b0;
        done          = 1'b0;
        
        case (state)
            ST_PASS1_MEAN: begin
                pe_opcode = OP_PASS_X;
                accum_en  = data_valid;
                if (data_valid && chunk_cnt == 5'd23) accum_fetch = 1'b1;
            end

            ST_LOAD_MEAN: begin
                pe_opcode = OP_LOAD_MEAN; 
            end

            ST_PASS2_VAR: begin
                pe_opcode = OP_VAR_SQR;
                accum_en  = data_valid;
                if (data_valid && chunk_cnt == 5'd23) accum_fetch = 1'b1;
            end

            ST_TRIG_SQRT: begin
                sqrt_valid_in = 1'b1; 
            end

            ST_CALC_SQRT: begin
                // All outputs remain at default 0 while waiting
                sqrt_busy = 1'b1;
            end

            ST_PASS3_NORM: begin
                pe_opcode = OP_NORMALIZE; 
                out_valid = data_valid;   
                
                // Assert done only when the final chunk of the final row is valid
                if (data_valid && chunk_cnt == 5'd23 && row_cnt == 9'd511) begin
                    done = 1'b1;
                end
            end
            
            default: begin
                pe_opcode = OP_PASS_X;
            end
        endcase
    end
endmodule