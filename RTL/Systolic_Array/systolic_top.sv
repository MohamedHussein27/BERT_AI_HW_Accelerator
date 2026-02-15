// assumptions 
// the fetch logic should provide zero input for N-1 cycles (with high valid in) so that we can drain the systolic
// last tile indicates if this systolic operation will produce a final tile result not partial sums
// so that this tile will be stored in a didcated buffer not the feedback buffer.
// the systolic array needs input of zeroes if there is no partial sums yet

module systolic_top #(
    parameter DATAWIDTH = 8,
    parameter DATAWIDTH_output = 32,
    parameter N_SIZE = 32,
    parameter num_of_raws = 512,
    parameter BUS_WIDTH = 256, // N_SIZE * DATAWIDTH
    parameter ADDR_WIDTH  = 10,
    parameter DEPTH = 543
) (
    input  logic [BUS_WIDTH-1:0] in_A,
    input  logic [(BUS_WIDTH*N_SIZE)-1:0] weights,
    input  logic clk,
    input  logic rst_n,
    input  logic valid_in,
    input  logic load_weight,
    input  logic last_tile,
    input  logic first_iteration 
    input  logic [ADDR_WIDTH-1:0] rd_addr_outbuffer,
    // Status signals
    output logic ready,
    output logic busy,
    output logic done,

    output logic [(DATAWIDTH_output*N_SIZE)-1:0] out_data_outbuffer // this is the output of the systolic buffer
);
// internal wires 
    logic sys_wt_en;
    logic we, we_outbuffer_wire;
    logic [ADDR_WIDTH-1:0] rd_addr;
    logic [ADDR_WIDTH-1:0] wr_addr;

    logic [DATAWIDTH-1:0] in_A_wire [N_SIZE-1:0];
    logic [(DATAWIDTH)-1:0] skew_out_wire[N_SIZE-1:0];

    logic [DATAWIDTH_output - 1:0] in_B_wire [N_SIZE-1:0];
    
    logic [(DATAWIDTH_output*N_SIZE)-1:0] interbuffer_output;
    logic [(DATAWIDTH_output*N_SIZE)-1:0] interbuffer_intput;
    logic [DATAWIDTH_output-1:0] out_C_wire [N_SIZE-1:0];


    assign we_outbuffer_wire = (last_tile) ? we : 0; 

    genvar i, j, k;
    generate
        for (i = 0; i < N_SIZE; i = i + 1 ) begin
            assign in_A_wire[i] = in_A[i*DATAWIDTH +: DATAWIDTH];
        end

        for (j = 0; j < N_SIZE; j = j + 1 ) begin
            assign in_B_wire[j] = (!first_iteration)? interbuffer_output[j*DATAWIDTH_output +: DATAWIDTH_output] : '0;
        end

        for (k = 0; k < N_SIZE; k = k + 1 ) begin
            assign interbuffer_intput[k*DATAWIDTH_output +: DATAWIDTH_output] = out_C_wire[k];
        end      
    endgenerate

    // systolic controller
    systolic_controller #(
        .DATAWIDTH(DATAWIDTH),
        .N_SIZE(N_SIZE),
        .num_of_raws(num_of_raws),
        .BUS_WIDTH(BUS_WIDTH), // N_SIZE * DATAWIDTH
        .ADDR_WIDTH(ADDR_WIDTH)
    ) contoller (
        .clk(clk),
        .rst_n(rst_n),
        .valid_in(valid_in),
        .load_weight(load_weight),
        .sys_wt_en(sys_wt_en),
        .we(we),
        .rd_addr(rd_addr),
        .wr_addr(wr_addr),
        .ready(ready),
        .busy(busy),
        .done(done)
    );
    // skew like input 
    skew_buffer #(
        .DATAWIDTH(DATAWIDTH),
        .N_SIZE(N_SIZE)
    ) u_skew_buffer (
        .clk      (clk),
        .rst_n    (rst_n),
        .valid_in (valid_in),
        .in_A     (in_A_wire),
        .out      (skew_out_wire)
    );
    // systolic array
    systolic #(
        .DATAWIDTH (DATAWIDTH),
        .DATAWIDTH_output(DATAWIDTH_output),
        .N_SIZE    (N_SIZE)
    ) u_systolic (
        .clk      (clk),
        .rst_n    (rst_n),
        .wt_en    (sys_wt_en),
        .valid_in (valid_in),
        .matrix_A (skew_out_wire),
        .matrix_B (in_B_wire),
        .wt_flat  (),// the weight input should be handeld TODO
        .matrix_C (out_C_wire)
    );
    // partial sum buffer
    systolic_internal_buffer #(
        .DATAWIDTH_output(DATAWIDTH_output),
        .N_SIZE(N_SIZE),
        .DEPTH     (DEPTH),
        .ADDR_WIDTH(ADDR_WIDTH)
    ) partial_sum_buffer (
        .clk     (clk),
        .we      (we),
        .rd_addr (rd_addr),
        .wr_addr (wr_addr),
        .in_data (interbuffer_intput), 
        .out_data(interbuffer_output)
    );
    // output buffer
    systolic_internal_buffer #(
        .DATAWIDTH (BUS_WIDTH),
        .DEPTH     (DEPTH),
        .ADDR_WIDTH(ADDR_WIDTH)
    ) out_buffer (
        .clk     (clk),
        .we      (we_outbuffer_wire),
        .rd_addr (rd_addr_outbuffer),
        .wr_addr (wr_addr),
        .in_data (interbuffer_intput),
        .out_data(out_data_outbuffer)
    );
endmodule