// this module deskew the systolic output in case of last_tile signal is high
// other wise this unit is off.

module deskew_buffer #(
    parameter DATAWIDTH_output = 32,
    parameter N_SIZE = 32
) (
    input logic clk,
    input logic rst_n,
    input logic valid_in,
    input logic signed [DATAWIDTH_output-1:0] data_in [N_SIZE-1:0],


    output logic signed [DATAWIDTH_output-1:0] data_out [N_SIZE-1:0]
);

localparam delay_bits = $clog2(N_SIZE);

genvar i;
generate
    for (i = 0; i < N_SIZE; i = i + 1) begin
        localparam [delay_bits-1:0] delay = N_SIZE-i-1;
        if (delay == 0) begin
            assign data_out[i] = data_in[i];
        end
        else begin
            logic signed [DATAWIDTH_output-1:0] delay_pipe [delay];
            always_ff @(posedge clk or negedge rst_n) begin
                if (!rst_n) begin
                    for (int j = 0; j < delay; j = j + 1) begin
                        delay_pipe[j] <= '0;
                    end
                end
                else begin
                    delay_pipe [0] <= (valid_in)? data_in[i] : delay_pipe [0];
                    for (int j = 1; j < delay; j = j + 1) begin
                        delay_pipe[j] <= (valid_in)? delay_pipe[j-1] : delay_pipe[j];
                    end
                end
            end
            assign data_out[i] = delay_pipe[delay-1];
        end
    end
endgenerate
    
endmodule