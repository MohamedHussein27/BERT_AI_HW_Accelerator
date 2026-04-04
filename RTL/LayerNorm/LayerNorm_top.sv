`timescale 1ns / 1ps
import PE_pkg::*;

module layernorm_top #(
    parameter DATAWIDTH = 32
)(
    input  logic        clk,
    input  logic        rst_n,

    // Interface from Main System Controller / Fetch Logic
    input  logic        data_valid, 
    input  logic signed [DATAWIDTH-1:0] buffer_rdata [0:31], // Assumed already Q5.26

    // Output to Next Layer (or Write-back Buffer)
    output logic signed [DATAWIDTH-1:0] norm_out_data [0:31],
    output logic        norm_out_valid,
    output logic        done,
    output logic        busy // mainly for square root module as it the one with most delay
);

    localparam logic signed [31:0] RECIP_768 = 32'd87381; // 1/768 in Q5.26
    localparam logic signed [31:0] EPSILON   = 32'd671;   // 0.00001 in Q5.26

    // Interconnect Wires
    pe_op_e             pe_opcode;
    logic signed [31:0] bcast_data;
    logic signed [31:0] pe_to_tree [0:31];
    logic signed [31:0] tree_out;
    logic signed [36:0] accum_out; // Increased bit width for safety
    
    logic accum_en, accum_fetch;
    logic sqrt_vin, sqrt_vout;
    logic signed [31:0] sqrt_result;

    // FSM (Now acts as a datapath coordinator)
    layernorm_fsm u_fsm (
        .clk(clk), 
        .rst_n(rst_n),
        .data_valid(data_valid),
        .pe_opcode(pe_opcode), 
        .accum_en(accum_en), 
        .accum_fetch(accum_fetch),
        .sqrt_valid_in(sqrt_vin), 
        .sqrt_valid_out(sqrt_vout),
        .sqrt_busy(busy),
        .out_valid(norm_out_valid), 
        .done(done)
    );

    // 32 Processing Elements
    genvar i;
    generate
        for (i = 0; i < 32; i++) begin : gen_pes
            processing_element u_pe (
                .clk(clk), 
                .rst_n(rst_n), 
                .opcode(pe_opcode),
                .data_in(buffer_rdata[i]), 
                .bcast_data(bcast_data),
                .data_out(pe_to_tree[i])
            );
            assign norm_out_data[i] = pe_to_tree[i]; 
        end
    endgenerate

    // Adder Tree & Accumulator
    adder_tree u_tree (
        .pe_data_in(pe_to_tree), 
        .tree_sum_out(tree_out)
    );

    accumulator #(
        .DATAWIDTH(32),
        .DATAWIDTH_OUTPUT(37) // Allows for summing 768 elements safely
    ) u_accum (
        .clk(clk), 
        .rst_n(rst_n), 
        .valid_in(accum_en), 
        .fetch(accum_fetch),
        .data_in(tree_out), 
        .data_out(accum_out)
    );

    // Scalar Math & Broadcast MUX
    logic signed [68:0] mul_recip;
    logic signed [31:0] accum_scaled;
    logic signed [31:0] var_plus_eps;

    always_comb begin
        // Multiply sum by 1/768 and scale back to Q5.26
        mul_recip    = accum_out * RECIP_768;
        accum_scaled = mul_recip >>> 26; 
        
        // Add numerical stability constant
        var_plus_eps = accum_scaled + EPSILON;

        // Broadcast MUX
        if (pe_opcode == OP_LOAD_MEAN) bcast_data = accum_scaled; 
        else                           bcast_data = sqrt_result;  
    end

    // Newton-Raphson Sqrt
    inv_sqrt u_sqrt (
        .clk(clk), 
        .rst_n(rst_n), 
        .valid_in(sqrt_vin),
        .data_in(var_plus_eps), 
        .data_out(sqrt_result), 
        .valid_out(sqrt_vout)
    );

endmodule