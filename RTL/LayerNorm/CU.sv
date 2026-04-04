`timescale 1ns / 1ps
import PE_pkg::*;

module layernorm_fsm (
    input  logic       clk,
    input  logic       rst_n,
    input  logic       data_valid, 
    output pe_op_e     pe_opcode,
    output logic       accum_en,
    output logic       accum_fetch,
    output logic       sqrt_valid_in,
    input  logic       sqrt_valid_out,
    output logic       sqrt_busy, 
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
    localparam ROW_CNT_WIDTH = $clog2(512);

    logic [1:0] load_parameters; 
    logic [4:0] chunk_cnt; 
    logic [ROW_CNT_WIDTH-1:0] row_cnt; 

    // Sequential Block
    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            state <= ST_PASS1_MEAN;
            chunk_cnt <= '0;
            row_cnt <= '0;
            load_parameters <= '0;
        end else begin
            state <= next_state;
            
            if (state == ST_PASS1_MEAN || state == ST_PASS2_VAR || state == ST_PASS3_NORM) begin
                if (data_valid) begin
                    if (chunk_cnt == 5'd23) chunk_cnt <= '0;
                    else                    chunk_cnt <= chunk_cnt + 1;
                end
            end else begin
                chunk_cnt <= '0; 
            end

            // Parameter loading handshake logic
            if (state == ST_CALC_SQRT && data_valid) begin
                if (load_parameters == 0)      load_parameters <= 1;
                else if (load_parameters == 1) load_parameters <= 2;
            end

            if (state == ST_PASS3_NORM && data_valid && chunk_cnt == 5'd23) begin
                row_cnt <= row_cnt + 1;
            end
        end
    end

    // State Transition Logic
    always_comb begin
        next_state = state;
        case (state)
            ST_PASS1_MEAN: if (data_valid && chunk_cnt == 5'd23) next_state = ST_LOAD_MEAN;
            ST_LOAD_MEAN:  next_state = ST_PASS2_VAR;
            ST_PASS2_VAR:  if (data_valid && chunk_cnt == 5'd23) next_state = ST_TRIG_SQRT;
            ST_TRIG_SQRT:  next_state = ST_CALC_SQRT;
            ST_CALC_SQRT:  if (sqrt_valid_out) next_state = ST_PASS3_NORM;
            ST_PASS3_NORM: if (data_valid && chunk_cnt == 5'd23) next_state = ST_PASS1_MEAN;
            default:       next_state = ST_PASS1_MEAN;
        endcase
    end

    // Combinational Output Logic
    always_comb begin
        pe_opcode     = OP_PASS_X;
        accum_en      = 1'b0;
        accum_fetch   = 1'b0;
        sqrt_valid_in = 1'b0;
        out_valid     = 1'b0;
        done          = 1'b0;
        sqrt_busy     = 1'b0; // Prevent Latch!

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
                sqrt_busy = 1'b1;
            end
            ST_CALC_SQRT: begin 
                if (data_valid && load_parameters == 0)      pe_opcode = OP_LOAD_WGT;
                else if (data_valid && load_parameters == 1) pe_opcode = OP_LOAD_BIAS;
                else                                         pe_opcode = OP_PASS_X;
                
                sqrt_busy = 1'b1;
            end
            ST_PASS3_NORM: begin
                pe_opcode = OP_NORMALIZE; 
                out_valid = data_valid;   
                if (data_valid && chunk_cnt == 5'd23) done = 1'b1;
            end
        endcase
    end
endmodule