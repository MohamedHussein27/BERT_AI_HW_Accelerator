// module to generate and connect PE's to each other
module systolic #(
    parameter DATAWIDTH = 8,
    parameter N_SIZE = 32
) (
    input logic clk,
    input logic rst_n,
    input logic wt_en,
    input logic [(DATAWIDTH) - 1:0] matrix_A [N_SIZE-1:0],   
    input logic [(DATAWIDTH*2) - 1:0] matrix_B [N_SIZE-1:0],
    input logic [DATAWIDTH-1:0] wt_flat [N_SIZE*N_SIZE-1:0],
    output logic [(DATAWIDTH*2) - 1:0] matrix_C [N_SIZE-1:0]
);
    // used to pass the elements row wise and column wise
    logic [DATAWIDTH-1:0] row_wire [0:N_SIZE];
    logic [(DATAWIDTH*2) - 1:0] col_wire [0:N_SIZE][0:N_SIZE];
    logic [DATAWIDTH-1:0] weight_wire [0:N_SIZE][0:N_SIZE];

    genvar l, p;
    generate
        for (l = 0; l < N_SIZE; l = l + 1) begin
            for (p = 0; p < N_SIZE; p = p + 1) begin
                always_comb begin
                    weight_wire[l][p] = wt_flat[l*N_SIZE + p];
                end
            end
        end
    endgenerate
    
    // feeding the matrices 
    genvar i;
    generate
        for (i = 0; i < N_SIZE; i = i + 1) begin
            always_comb begin
                row_wire[i] = matrix_A[i];
                col_wire[0][i] = matrix_B[i];
            end
        end
    endgenerate
    // instantiation of PE's
    genvar ii, jj;
    generate
        for (ii = 0; ii < N_SIZE; ii = ii + 1) begin // row loop
            for (jj = 0; jj < N_SIZE; jj = jj + 1) begin // column loop
                PE #(.DATAWIDTH(DATAWIDTH)) pe_inst(
                    .clk(clk),
                    .rst_n(rst_n),
                    .wt_en(wt_en),
                    .wt(weight_wire[ii][jj]),
                    .in_A(row_wire[ii]),
                    .in_B(col_wire[ii][jj]),
                    .out_D(col_wire[ii+1][jj])
                );
            end
        end
    endgenerate

    // output raw should be stores in the buffer.
    //TODO --> modify the intermidiate buffers to carry the an output of size 576*32 as every raw produce 64 trash value.
    // i  can use this cycles to store the outtile in the format to be ready to be inputted next systolic operation.
    // insert zero at the trash values.
    genvar k;
    generate
        for (k = 0; k < N_SIZE; k = k + 1) begin
            always_comb begin
                matrix_C[k] = col_wire[N_SIZE][k];
            end
        end
    endgenerate
endmodule