`timescale 1ns / 1ps

import PE_pkg::*;

module processing_element #(
    parameter Q_FRAC_BITS = 26,
    parameter DATA_WIDTH  = 32
)(
    input  logic                   clk,
    input  logic                   rst_n,
    
    // Control and Data Inputs
    input  pe_op_e                 opcode,
    input  logic signed [31:0]     data_in,      // x_i from the buffer
    input  logic signed [31:0]     bcast_data,   // Broadcasted scalar
    
    // Output to Adder Tree or Buffer
    output logic signed [31:0]     data_out
);

    // --------------------------------------------------------
    // Local Memory (Registers for Gamma, Beta, and Mean)
    // --------------------------------------------------------
    logic signed [31:0] local_weight_reg; // Gamma
    logic signed [31:0] local_bias_reg;   // Beta
    logic signed [31:0] local_mean_reg;   // Mean for normalization

    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            local_weight_reg <= '0;
            local_bias_reg   <= '0;
            local_mean_reg   <= '0;
        end else begin
            if (opcode == OP_LOAD_WGT)  local_weight_reg <= data_in;
            if (opcode == OP_LOAD_BIAS) local_bias_reg   <= data_in;
            if (opcode == OP_LOAD_MEAN) local_mean_reg   <= bcast_data;
        end
    end

    // --------------------------------------------------------
    // Combinational Datapath (Subtractor -> Multiplier -> Adder)
    // --------------------------------------------------------
    logic signed [31:0] sub_out;
    logic signed [63:0] mul_full; 
    logic signed [31:0] mul_out;
    
    always_comb begin
        // ==========================================
        // Stage 1: The Subtractor (SUB)
        // ==========================================
        if (opcode == OP_VAR_SQR || opcode == OP_NORMALIZE) begin
            sub_out = data_in - local_mean_reg; // (X - mu)
        end else begin
            sub_out = data_in; // Bypass subtractor
        end

        // ==========================================
        // Stage 2: The Multiplier (MUL) 
        // ==========================================
        case (opcode)
            OP_VAR_SQR:   mul_full = sub_out * sub_out;       
            OP_NORMALIZE: mul_full = sub_out * bcast_data;     
            default:      mul_full = {32'd0, sub_out};        // Pad bypassed data
        endcase
        
        // Rounding, Shifting, and Clamping
        if (opcode == OP_VAR_SQR || opcode == OP_NORMALIZE) begin
            mul_out = quantize_32(mul_full);
        end else begin
            mul_out = mul_full[31:0]; // Bypass case
        end

        // ==========================================
        // Stage 3: The Adder (ADD) & Output Routing
        // ==========================================
        if (opcode == OP_NORMALIZE) begin
            mul_full = mul_out * local_weight_reg; 
            mul_out  = quantize_32(mul_full);
            data_out = mul_out + local_bias_reg; // Add Beta
        end else begin
            data_out = mul_out; // Bypass Adder
        end
    end

    function logic signed [31:0] quantize_32 (
        input logic signed [63:0] mul_full
    );
        logic signed [63:0] mul_rounded;
        // Add half (2^25) for rounding, then shift right by 26
        mul_rounded = (mul_full + (64'(1) << (Q_FRAC_BITS - 1))) >>> Q_FRAC_BITS;
        
        // Saturation Clamping (Protect against integer overflow)
        if (mul_rounded > 64'sh000000007FFFFFFF) begin
            return 32'sh7FFFFFFF; // Clamp to max positive
        end else if (mul_rounded < -64'sh0000000080000000) begin
            return -32'sh80000000; // Clamp to max negative
        end else begin
            return mul_rounded[31:0]; // Safe to take bottom 32 bits
        end
    endfunction
        
endmodule