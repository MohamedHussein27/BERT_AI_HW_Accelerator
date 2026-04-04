module accumulator #(
    parameter DATAWIDTH_IN = 42,    
    parameter DATAWIDTH_OUTPUT = 47  
) (
    input logic clk,
    input logic rst_n,
    input logic valid_in, 
    input logic fetch,    
    input logic signed [DATAWIDTH_IN-1:0] data_in,
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
                data_out <= acc_reg + $signed(data_in);
                acc_reg  <= '0; 
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