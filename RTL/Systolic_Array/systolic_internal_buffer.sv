module systolic_internal_buffer #(
    parameter DATAWIDTH_output = 32,
    parameter N_SIZE = 32,
    parameter DEPTH = 543, // (512 + 31(zeros)) 
    parameter ADDR_WIDTH  = 10
) (
    input logic clk,
    input logic we,
    input logic [ADDR_WIDTH-1:0] rd_addr,
    input logic [ADDR_WIDTH-1:0] wr_addr,
    input logic [(DATAWIDTH_output*N_SIZE)-1:0] in_data,

    output logic [(DATAWIDTH_output*N_SIZE)-1:0] out_data
);

    logic [(DATAWIDTH_output*N_SIZE)-1:0] mem [DEPTH-1:0];

    always @(posedge clk) begin
        if (we) begin
            mem[wr_addr] <= in_data;
        end
        out_data <= mem[rd_addr];
    end
endmodule