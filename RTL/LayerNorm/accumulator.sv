module accumulator #(
    parameter DATAWIDTH = 32, // input Q5.26
    parameter DATAWIDTH_OUTPUT = 36 // output Q9.26
) (
    input logic clk,
    input logic rst_n,
    input logic valid_in, // indicates if we are going to accumlate or no.
    input logic fetch,    // indicates if we want to read the accumlated result or no (reg is cleared after it).
    input logic signed [DATAWIDTH-1:0] data_in,

    output logic signed [DATAWIDTH_OUTPUT-1:0] data_out  

);

    logic signed [DATAWIDTH_OUTPUT-1:0] acc_reg;
    always_ff @(posedge clk or negedge rst_n) begin 
        if (!rst_n) begin
            data_out <= '0;
            acc_reg  <= '0;
        end 
        else begin
            if (fetch && valid_in) begin
                // Send the completed sum to output
                data_out <= acc_reg; 
                // Start the NEW sum immediately with the current data
                acc_reg  <= $signed(data_in); 
            end
            else if (fetch) begin
                data_out <= acc_reg;
                acc_reg  <= '0;
            end
            else if (valid_in) begin
                acc_reg <= acc_reg + $signed(data_in);
            end
        end
    end
endmodule