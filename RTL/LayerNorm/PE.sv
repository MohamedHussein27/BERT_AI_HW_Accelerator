`timescale 1ns / 1ps
import PE_pkg::*;

module processing_element #(
    parameter Q_FRAC_BITS = 26,
    parameter DATA_WIDTH  = 32
)(
    input  logic                   clk,
    input  logic                   rst_n,
    input  pe_op_e                 opcode,
    input  logic signed [31:0]     data_in,      // Input precision: Q5.26 (1 sign, 5 int, 26 frac)
    input  logic signed [31:0]     bcast_data,   // Input precision: Q5.26 
    
    output logic signed [36:0]     data_out      // Output precision: Q10.26 (1 sign, 10 int, 26 frac)
);

    logic signed [31:0] local_weight_reg; // Precision: Q5.26
    logic signed [31:0] local_bias_reg;   // Precision: Q5.26
    logic signed [31:0] local_mean_reg;   // Precision: Q5.26

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

    logic signed [36:0] sub_out;  // Q10.26
    logic signed [73:0] mul_full; // Q21.52
    logic signed [36:0] mul_out;  // Q10.26

    // Stage 1: Subtractor (Cast inputs to 37 bits first to prevent overflow)
    always_comb begin
        if (opcode == OP_VAR_SQR || opcode == OP_NORMALIZE) begin 
            // Q10.26
            sub_out = 37'(data_in) - 37'(local_mean_reg);
        end else begin
            // 37'(Q5.26) becomes Q10.26
            sub_out = 37'(data_in); 
        end
    end

    // Stage 3: Multipliers and Adders
    always_comb begin
        case (opcode)
            // Multiplication: Q10.26 * Q10.26 = Q21.52 (74 bits)
            OP_VAR_SQR:   mul_full = sub_out * sub_out; // result is Q21.52
            
            // Multiplication: Q10.26 * 37'(Q5.26 -> Q10.26) = Q21.52
            OP_NORMALIZE: mul_full = sub_out * 37'(bcast_data); 
            
            // Bypass shift: Cast Q10.26 to 74 bits (physically Q47.26). 
            // Left shift by 26 moves the fractional bits up, making it exactly Q21.52! 
            // This perfectly aligns the data so quantize_37 can shift it back down safely.
            default:      mul_full = 74'(sub_out) <<< Q_FRAC_BITS; // bypass shift ( Q47.26 -> Q21.52 )
        endcase
        
        if (opcode == OP_VAR_SQR || opcode == OP_NORMALIZE) begin
            // Quantize function scales Q21.52 back down to Q10.26
            mul_out = quantize_37(mul_full);
        end else begin
            // Bypass logic natively passes the Q10.26 sub_out
            mul_out = sub_out; 
        end

        if (opcode == OP_NORMALIZE) begin
            // Q10.26 * 37'(Q5.26 -> Q10.26) = Q21.52
            mul_full = mul_out * 37'(local_weight_reg); 
            
            // Quantize function scales Q21.52 back down to Q10.26
            mul_out  = quantize_37(mul_full);
            
            // Addition: Q10.26 + 37'(Q5.26 -> Q10.26) = Q10.26
            data_out = mul_out + 37'(local_bias_reg); 
        end else begin
            // Natively passes Q10.26
            data_out = mul_out; 
        end
    end

    // Stage 2: Quantize 74-bit multiplication down to 37 bits
    function logic signed [36:0] quantize_37 (
        input logic signed [73:0] mul_full_in // Input is Q21.52
    );
        logic signed [73:0] mul_rounded;
        
        // 1. Adds `1 << 25` (which equals 0.5 in the discarded 26-bit fraction) for round-half-up.
        // 2. Arithmetic shift right (>>>) by 26.
        // Precision physically remains 74 bits, but mathematically drops from Q21.52 to Q21.26.
        mul_rounded = (mul_full_in + (74'(1) << (Q_FRAC_BITS - 1))) >>> Q_FRAC_BITS;
        
        // Saturation bounds for 37-bit (Q10.26)
        // If the remaining integer part (Q21) exceeds the 10 integer bits allowed in Q10.26, clamp it.
        if (mul_rounded > 74'sh0000000FFFFFFFFF) begin
            return 37'h0FFFFFFFFF; // Clamped to Max Positive Q10.26
        end else if (mul_rounded < -74'sh0000001000000000) begin
            return 37'h1000000000; // Clamped to Max Negative Q10.26
        end else begin
            // Safely fits, truncate the top 37 redundant sign-extension bits
            return mul_rounded[36:0]; 
        end
    endfunction
endmodule