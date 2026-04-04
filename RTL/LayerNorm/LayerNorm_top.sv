`timescale 1ns / 1ps
import PE_pkg::*;

module layernorm_top #(
    parameter DATAWIDTH = 32
)(
    input  logic        clk,
    input  logic        rst_n,
    input  logic        data_valid, 
    input  logic signed [DATAWIDTH-1:0] buffer_rdata [0:31], 
    output logic signed [DATAWIDTH-1:0] norm_out_data [0:31],
    output logic        norm_out_valid,
    output logic        done,
    output logic        busy 
);

    localparam logic signed [31:0] RECIP_768 = 32'd87381; 
    localparam logic signed [31:0] EPSILON   = 32'd671;   

    pe_op_e             pe_opcode;
    logic signed [31:0] bcast_data;
    logic signed [36:0] pe_to_tree [0:31];
    
    logic signed [41:0] tree_out;  
    logic signed [46:0] accum_out; 
    
    logic accum_en, accum_fetch;
    logic sqrt_vin, sqrt_vout;
    logic signed [31:0] sqrt_result;

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
            // Safely truncate normalized result back to standard 32-bit for next layer
            assign norm_out_data[i] = pe_to_tree[i][31:0]; 
        end
    endgenerate

    adder_tree #(
        .NUM_INPUTS(32),
        .DATA_WIDTH_IN(37),
        .DATA_WIDTH_OUT(42)
    ) u_tree (
        .pe_data_in(pe_to_tree), 
        .tree_sum_out(tree_out)
    );

    accumulator #(
        .DATAWIDTH_IN(42),
        .DATAWIDTH_OUTPUT(47) 
    ) u_accum (
        .clk(clk), 
        .rst_n(rst_n), 
        .valid_in(accum_en), 
        .fetch(accum_fetch),
        .data_in(tree_out), 
        .data_out(accum_out)
    );

    // 47 bits * 32 bits = 79 bits required to prevent multiplication overflow
    logic signed [78:0] mul_recip; 
    logic signed [31:0] accum_scaled;
    logic signed [31:0] var_plus_eps;

    always_comb begin
        mul_recip    = accum_out * RECIP_768;
        
        // Truncate down to 32 bits. Since we divided by 768, the massive sum 
        // shrinks back down to the ~23.0 variance, which safely fits in Q5.26!
        accum_scaled = 32'(mul_recip >>> 26); 
        
        var_plus_eps = accum_scaled + EPSILON;
        
        if (pe_opcode == OP_LOAD_MEAN) bcast_data = accum_scaled;
        else                           bcast_data = sqrt_result;
    end

    inv_sqrt u_sqrt (
        .clk(clk), 
        .rst_n(rst_n), 
        .valid_in(sqrt_vin),
        .data_in(var_plus_eps), 
        .data_out(sqrt_result), 
        .valid_out(sqrt_vout)
    );

endmodule