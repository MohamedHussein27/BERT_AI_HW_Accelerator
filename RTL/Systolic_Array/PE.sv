module PE #(
    parameter DATAWIDTH = 8
)(
    input  logic clk,
    input  logic rst_n,
    input  logic wt_en,
    input  logic valid_in,

    input  logic [(DATAWIDTH) - 1:0] wt,
    input  logic [(DATAWIDTH) - 1:0] in_A,
    input  logic [(DATAWIDTH*3) - 1:0] in_B,

    output logic [(DATAWIDTH*3) - 1:0] out_D
);

    logic [(DATAWIDTH) - 1:0] weight;

    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            out_D  <= 0;
            weight <= 0;
        end else begin
            // load weight into local reg
            if (wt_en) 
                weight <= wt;

            // only compute when valid
            if (valid_in)
                (* use_dsp = "yes" *) out_D <= (in_A * weight) + in_B;
        end
    end
endmodule