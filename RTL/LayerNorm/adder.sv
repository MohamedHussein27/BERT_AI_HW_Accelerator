module adder #(
    parameter DATAWIDTH = 32,
    parameter FRAC_BITS = 26
) (
    input  logic                          clk,
    input  logic                          rst_n,
    input  logic                          valid_in,
    input  logic signed [DATAWIDTH-1:0]   data_in_1,
    input  logic signed [DATAWIDTH-1:0]   data_in_2,
    output logic signed [DATAWIDTH-1:0]   data_out
);

    localparam signed [DATAWIDTH-1:0] MAX_VAL =  (2**(DATAWIDTH-1))-1;  
    localparam signed [DATAWIDTH-1:0] MIN_VAL = -(2**(DATAWIDTH-1)); 

    logic signed [DATAWIDTH:0] full_sum;
    assign full_sum = data_in_1 + data_in_2;

    logic ov_pos, ov_neg;
    assign ov_pos = ~data_in_1[DATAWIDTH-1] & ~data_in_2[DATAWIDTH-1]
                  &  full_sum[DATAWIDTH-1];
    assign ov_neg =  data_in_1[DATAWIDTH-1] &  data_in_2[DATAWIDTH-1]
                  & ~full_sum[DATAWIDTH-1];

    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            data_out <= '0;
        end else begin
            if (valid_in) begin
                if      (ov_pos) data_out <= MAX_VAL;
                else if (ov_neg) data_out <= MIN_VAL;
                else             data_out <= full_sum[DATAWIDTH-1:0];
            end
        end
    end

endmodule