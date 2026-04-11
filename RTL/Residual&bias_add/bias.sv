module bias #(
    parameter  DATAWIDTH = 8,
    parameter  VEC_SIZE = 32
) (
    input  logic rst_n,
    input  logic signed [VEC_SIZE-1:0][DATAWIDTH-1:0] data_in_1,
    input  logic signed [VEC_SIZE-1:0][DATAWIDTH-1:0] data_in_2,
    output logic signed [VEC_SIZE-1:0][DATAWIDTH-1:0] data_out
);

    localparam signed [DATAWIDTH-1:0] MAX_VAL =  (1 << (DATAWIDTH-1)) - 1;
    localparam signed [DATAWIDTH-1:0] MIN_VAL = -(1 << (DATAWIDTH-1));

    always_comb begin
        for (int i = 0; i < VEC_SIZE; i++) begin
            logic signed [DATAWIDTH:0] ext_sum; 
            
            ext_sum = data_in_1[i] + data_in_2[i];
            // Standard > and < operators can now safely detect saturation
            if (ext_sum > MAX_VAL) begin
                data_out[i] = MAX_VAL;
            end 
            else if (ext_sum < MIN_VAL) begin
                data_out[i] = MIN_VAL;
            end 
            else begin
                data_out[i] = ext_sum[DATAWIDTH-1:0]; // Truncate back to original width
            end
        end
    end
endmodule