`timescale 1ns / 1ps

module vector_quantize #(
    parameter VEC_SIZE      = 32,  // Number of parallel elements
    parameter DATAWIDTH_in  = 8,   // 8-bit reading from buffer
    parameter DATAWIDTH_out = 32,  // 32-bit Q5.26 output to LayerNorm PEs
    parameter M_width       = 32,
    parameter S_width       = 8
)(
    input  logic                                         clk,
    input  logic                                         rst_n,
    input  logic                                         valid_in,
    
    // 2D Packed Array: Slices a flat 256-bit bus into 32 distinct 8-bit elements
    input  logic signed [VEC_SIZE-1:0][DATAWIDTH_in-1:0] data_in,
    
    // Shared parameters from your Distributed ROM
    input  logic        [M_width-1:0]                    scale_M,
    input  logic signed [S_width-1:0]                    scale_S,

    // 2D Packed Array: Slices a flat 1024-bit bus into 32 distinct 32-bit elements
    output logic signed [VEC_SIZE-1:0][DATAWIDTH_out-1:0] data_out,
    output logic                                         valid_out
);

    // Internal valid signals from each lane
    logic [VEC_SIZE-1:0] valid_out_lanes;
    
    // Since all 32 lanes operate in perfect parallel lockstep, 
    // we only need to look at lane 0 to know when the whole vector is valid.
    assign valid_out = valid_out_lanes[0];

    // Hardware Generate Loop: Stamps out 32 parallel quantizers
    genvar i;
    generate
        for (i = 0; i < VEC_SIZE; i++) begin : gen_quant_lanes
            quantize #(
                .DATAWIDTH_in(DATAWIDTH_in),
                .DATAWIDTH_out(DATAWIDTH_out),
                .M_width(M_width),
                .S_width(S_width)
            ) scalar_quant_inst (
                .clk(clk),
                .rst_n(rst_n),
                .valid_in(valid_in),
                .data_in(data_in[i]),      // Route 8-bit element 'i' into this lane
                .scale_M(scale_M),         // Shared scale multiplier
                .scale_S(scale_S),         // Shared scale shift
                .data_out(data_out[i]),    // Route 32-bit output 'i' out of this lane
                .valid_out(valid_out_lanes[i])
            );
        end
    endgenerate

endmodule