`timescale 1ns / 1ps
import PE_pkg::*;

module processing_element #(
    parameter Q_FRAC_BITS = 26,
    parameter DATA_WIDTH  = 32
)(
    input  logic                   clk,
    input  logic                   rst_n,
    input  pe_op_e                 opcode,
    input  logic signed [31:0]     data_in,      
    input  logic signed [31:0]     bcast_data,   
    
    // EXPANDED: 37 bits prevents (X-mu)^2 from clamping at 31.99
    output logic signed [36:0]     data_out 
);

    logic signed [31:0] local_weight_reg; 
    logic signed [31:0] local_bias_reg;   
    logic signed [31:0] local_mean_reg;   

    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            local_weight_reg <= '0;
            local_bias_reg   <= '0;
            local_mean_reg   <= '0;
        end else begin
            if (opcode == OP_LOAD_WGT)       local_weight_reg <= data_in;
            else if (opcode == OP_LOAD_BIAS) local_bias_reg   <= data_in;
            else if (opcode == OP_LOAD_MEAN) local_mean_reg   <= bcast_data;
        end
    end

    logic signed [36:0] sub_out;
    logic signed [73:0] mul_full; // 37 bits * 37 bits = 74 bits
    logic signed [36:0] mul_out;

    // Stage 1: Subtractor (Cast inputs to 37 bits first to prevent overflow)
    always_comb begin
        if (opcode == OP_VAR_SQR || opcode == OP_NORMALIZE) begin
            sub_out = 37'(data_in) - 37'(local_mean_reg);
        end else begin
            sub_out = 37'(data_in); 
        end
    end

    // Stage 3: Multipliers and Adders
    always_comb begin
        case (opcode)
            OP_VAR_SQR:   mul_full = sub_out * sub_out; 
            OP_NORMALIZE: mul_full = sub_out * 37'(bcast_data); 
            default:      mul_full = 74'(sub_out) <<< Q_FRAC_BITS; // bypass shift
        endcase
        
        if (opcode == OP_VAR_SQR || opcode == OP_NORMALIZE) begin
            mul_out = quantize_37(mul_full);
        end else begin
            mul_out = sub_out; 
        end

        if (opcode == OP_NORMALIZE) begin
            mul_full = mul_out * 37'(local_weight_reg); 
            mul_out  = quantize_37(mul_full);
            data_out = mul_out + 37'(local_bias_reg); 
        end else begin
            data_out = mul_out; 
        end
    end

    // Stage 2: Quantize 74-bit multiplication down to 37 bits
    function logic signed [36:0] quantize_37 (
        input logic signed [73:0] mul_full_in
    );
        logic signed [73:0] mul_rounded;
        mul_rounded = (mul_full_in + (74'(1) << (Q_FRAC_BITS - 1))) >>> Q_FRAC_BITS;
        
        // Saturation bounds for 37-bit (Q10.26)
        if (mul_rounded > 74'sh0000000FFFFFFFFF) begin
            return 37'h0FFFFFFFFF;
        end else if (mul_rounded < -74'sh0000001000000000) begin
            return 37'h1000000000;
        end else begin
            return mul_rounded[36:0];
        end
    endfunction
endmodule