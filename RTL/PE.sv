// wt --> input weight
// in_A left input, in_B up input
//        in_B
//          |
// in_A --> PE --> out_R
//          |
//        out_D

module PE #(
    parameter DATAWIDTH = 8
)(
    input logic clk,
    input logic rst_n,
    input logic wt_en,
    input logic [(DATAWIDTH) - 1:0] wt,
    input logic [(DATAWIDTH) - 1:0] in_A,
    input logic [(DATAWIDTH*2)-1:0] in_B,

    output logic [(DATAWIDTH) - 1:0] out_R,
    output logic [(DATAWIDTH*2)-1:0] out_D
);

    logic [(DATAWIDTH) - 1:0] weight;

    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            out_R  <= 0;
            out_D  <= 0;
            weight <= 0;
        end else begin
            // load weight into local reg
            if (wt_en) weight <= wt;

            out_D <= (in_A * weight) + in_B;  
            out_R <= in_A;  // pass data to the right
        end
    end
endmodule