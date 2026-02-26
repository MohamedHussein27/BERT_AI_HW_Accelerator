module PE #(
    parameter DATAWIDTH        = 8,
    parameter DATAWIDTH_output = 32
)(
    input  logic clk,
    input  logic rst_n,
    input  logic wt_en,
    input  logic valid_in,

    input  logic signed [(DATAWIDTH) - 1:0]    wt,
    input  logic signed [(DATAWIDTH) - 1:0]    in_A,
    input  logic signed [DATAWIDTH_output-1:0] in_B,

    output logic signed [DATAWIDTH_output-1:0] out_D,
    output logic signed [(DATAWIDTH) - 1:0]    out_R
);

    (* use_dsp = "yes" *) logic signed [DATAWIDTH_output-1:0] mac_out;

    logic signed [(DATAWIDTH)-1:0] weight;

    // Synchronous reset — allows the accumulator register to live
    // inside the DSP48 P-register (async reset blocks DSP inference)
    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            mac_out <= '0;
            out_R   <= '0;
            weight  <= '0;
        end else begin
            if (wt_en)
                weight <= wt;

            if (valid_in) begin
                mac_out <= (in_A * weight) + in_B;  // line unchanged
                out_R   <= in_A;
            end
        end
    end

    assign out_D = mac_out;

endmodule