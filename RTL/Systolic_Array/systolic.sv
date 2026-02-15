// module to generate and connect PE's to each other
module systolic #(
    parameter DATAWIDTH = 8,
    parameter DATAWIDTH_output = 32,
    parameter N_SIZE = 32
) (
    input logic clk,
    input logic rst_n,
    input logic wt_en,
    input logic valid_in,
    input logic [(DATAWIDTH) - 1:0] matrix_A [N_SIZE-1:0],   
    input logic [DATAWIDTH_output - 1:0] matrix_B [N_SIZE-1:0],
    input logic [DATAWIDTH-1:0] wt_serial [N_SIZE-1:0],
    input logic [$clog2(N_SIZE)-1:0] wt_row_sel,

    output logic [DATAWIDTH_output-1:0] matrix_C [N_SIZE-1:0]
);
    // used to pass the elements row wise and column wise
    logic [DATAWIDTH-1:0] row_wire [0:N_SIZE][0:N_SIZE];
    logic [DATAWIDTH_output-1:0] col_wire [0:N_SIZE][0:N_SIZE];

    logic wt_en_row [N_SIZE-1:0];

    genvar l;
    generate
        for (l = 0; l < N_SIZE; l = l + 1) begin
            assign wt_en_row[l] = wt_en && (wt_row_sel == l);
        end
    endgenerate
    
    // feeding the matrices 
    genvar i;
    generate
        for (i = 0; i < N_SIZE; i = i + 1) begin
            assign row_wire[i][0] = matrix_A[i];
            assign col_wire[0][i] = matrix_B[i];

        end
    endgenerate
    // instantiation of PE's
    genvar ii, jj;
    generate
        for (ii = 0; ii < N_SIZE; ii = ii + 1) begin // row loop
            for (jj = 0; jj < N_SIZE; jj = jj + 1) begin // column loop
                PE #(
                    .DATAWIDTH(DATAWIDTH),
                    .DATAWIDTH_output(DATAWIDTH_output))
                    pe_inst(
                    .clk(clk),
                    .rst_n(rst_n),
                    .wt_en(wt_en_row[ii]),
                    .valid_in(valid_in),
                    .wt(wt_serial[jj]),
                    .in_A(row_wire[ii][jj]),
                    .in_B(col_wire[ii][jj]),
                    .out_D(col_wire[ii+1][jj]),
                    .out_R(row_wire[ii][jj+1])
                );
            end
        end
    endgenerate

// a valid output is out after N cycles. and continue to (output for number_of_raws of the input tile) cycles
    genvar k;
    generate
        for (k = 0; k < N_SIZE; k = k + 1) begin
            assign matrix_C[k] = col_wire[N_SIZE][k];
        end
    endgenerate
endmodule