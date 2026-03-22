module adder_tree #(
    parameter DATAWIDTH = 32,
    parameter FRAC_BITS = 26,
    parameter NUM_OF_INPUTS = 32 
    // this number means that the first raw of the adder tree should contain  16 adder then 8 then 4-2-1.   
) (
    input logic clk,
    input logic rst_n,
    input logic valid_in,
    input logic signed [DATAWIDTH-1:0] data_in [NUM_OF_INPUTS-1:0],

    output logic signed[DATAWIDTH-1:0] result,
    output logic valid_out
);  

    logic valid_in_st1;
    logic valid_in_st2;
    logic valid_in_st3;
    logic valid_in_st4;
    logic valid_in_st5;

    always_ff @(posedge clk or negedge rst_n) begin : valid_out_handle
        if (!rst_n) begin
            valid_in_st1 <= 0;
            valid_in_st2 <= 0;
            valid_in_st3 <= 0;
            valid_in_st4 <= 0;
            valid_in_st5 <= 0;
        end
        else begin
            valid_in_st1 <= valid_in;
            valid_in_st2 <= valid_in_st1;
            valid_in_st3 <= valid_in_st2;
            valid_in_st4 <= valid_in_st3;
            valid_in_st5 <= valid_in_st4;
        end
    end

    logic signed [DATAWIDTH-1:0] data_out_st1[15:0];
    logic signed [DATAWIDTH-1:0] data_out_st2[7:0];
    logic signed [DATAWIDTH-1:0] data_out_st3[3:0];
    logic signed [DATAWIDTH-1:0] data_out_st4[1:0];
    logic signed [DATAWIDTH-1:0] data_out_st5;

    genvar i;
    generate
        // stage 1
        for (i = 0; i < NUM_OF_INPUTS/2; i++) begin
            adder #(
                .DATAWIDTH(DATAWIDTH),
                .FRAC_BITS(FRAC_BITS)
            ) adder_inst(
                .clk(clk),
                .rst_n(rst_n),
                .valid_in(valid_in),
                .data_in_1(data_in[2*i]),
                .data_in_2(data_in[2*i+1]),
                .data_out(data_out_st1[i])
            );
        end

        // stage 2
        for (i = 0; i < NUM_OF_INPUTS/4; i++) begin 
            adder #(
                .DATAWIDTH(DATAWIDTH),
                .FRAC_BITS(FRAC_BITS)
            ) adder_inst(
                .clk(clk),
                .rst_n(rst_n),
                .valid_in(valid_in_st1),
                .data_in_1(data_out_st1[2*i]),
                .data_in_2(data_out_st1[2*i+1]),
                .data_out(data_out_st2[i])
            );
        end

        // stage 3
        for (i = 0; i < NUM_OF_INPUTS/8; i++) begin 
            adder #(
                .DATAWIDTH(DATAWIDTH),
                .FRAC_BITS(FRAC_BITS)
            ) adder_inst(
                .clk(clk),
                .rst_n(rst_n),
                .valid_in(valid_in_st2),
                .data_in_1(data_out_st2[2*i]),
                .data_in_2(data_out_st2[2*i+1]),
                .data_out(data_out_st3[i])
            );
        end

        //stage 4 
        for (i = 0; i < NUM_OF_INPUTS/16; i++) begin 
            adder #(
                .DATAWIDTH(DATAWIDTH),
                .FRAC_BITS(FRAC_BITS)
            ) adder_inst(
                .clk(clk),
                .rst_n(rst_n),
                .valid_in(valid_in_st3),
                .data_in_1(data_out_st3[2*i]),
                .data_in_2(data_out_st3[2*i+1]),
                .data_out(data_out_st4[i])
            );
        end
        // stage 5
        for (i = 0; i < NUM_OF_INPUTS/32; i++) begin 
            adder #(
                .DATAWIDTH(DATAWIDTH),
                .FRAC_BITS(FRAC_BITS)
            ) adder_inst(
                .clk(clk),
                .rst_n(rst_n),
                .valid_in(valid_in_st4),
                .data_in_1(data_out_st4[2*i]),
                .data_in_2(data_out_st4[2*i+1]),
                .data_out(data_out_st5)
            );
        end
    endgenerate

    assign result = data_out_st5;
    assign valid_out = valid_in_st5;
endmodule