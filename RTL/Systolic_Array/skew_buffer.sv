// this is the module where i generate the delay chain requierd for the skewd like input for the systolic
module skew_buffer #(
    parameter DATAWIDTH = 8,
    parameter N_SIZE = 32
) (
    input logic clk,
    input logic rst_n,
    input logic valid_in, // output 0 if no valid input and hold registers values
    input logic [(DATAWIDTH)-1:0] in_A[N_SIZE],
    output logic [(DATAWIDTH)-1:0] out[N_SIZE]
);
    genvar i;
    generate
        for (i = 0; i < N_SIZE; i = i + 1 ) begin
            if (i == 0) begin // first raw do not have a delay
                // push 0 if no valid data.
                assign out[i] = (valid_in) ? in_A[i] : '0;
            end
            else begin
                // every raw need i num of delays (flipflops)
                logic [(DATAWIDTH)-1:0] delay_pipe [i+1];
                
                always_ff @(posedge clk or negedge rst_n) begin
                    if (!rst_n)begin
                        for (int j = 0; j < i+1; j++) begin
                            delay_pipe[j] <= '0;
                        end
                    end
                    else begin
                        delay_pipe[0] <= (valid_in) ? in_A[i] : delay_pipe[0];
                        for (int j = 1; j < i+1; j = j + 1) begin
                            // move the input along the chain
                            delay_pipe[j] <= (valid_in) ? delay_pipe[j-1] : delay_pipe[j];
                        end
                    end
                end
                assign out[i] = (valid_in) ? delay_pipe[i] : '0;
            end
        end
    endgenerate
endmodule