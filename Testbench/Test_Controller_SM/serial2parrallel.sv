module serial2parrallel #(
    parameter VEC_SIZE = 32,
    parameter ELEMENT_DATAWIDTH = 8
) (
    input logic clk,
    input logic rst_n,
    input logic valid_in,
    input logic [ELEMENT_DATAWIDTH-1:0] data_in,

    output logic [ELEMENT_DATAWIDTH*VEC_SIZE-1:0] data_out,
    output logic valid_out
);

    localparam COUNTER_WIDTH = $clog2(VEC_SIZE);

    logic [VEC_SIZE-1:0][ELEMENT_DATAWIDTH-1:0] data_reg;
    logic [COUNTER_WIDTH-1:0] count;

    always_ff @(posedge clk or negedge rst_n ) begin : main_block
        if (!rst_n) begin
            data_reg  <= '0;
            data_out  <= '0;
            valid_out <= 1'b0;
        end
        else begin
            valid_out <= 1'b0; // defult

            if (valid_in) begin
                data_reg[count] <= data_in;
                if (count == VEC_SIZE - 1) begin
                    data_out <= {data_in, data_reg[VEC_SIZE-2:0]};
                    valid_out <= 1;
                end
            end

        end
    end   


    always_ff @( posedge clk or negedge rst_n ) begin : counter_block
        if (!rst_n)begin
            count <= '0;
        end
        else begin
            if (valid_in) begin
                count <= count + 1;
                if (count == VEC_SIZE - 1) begin
                    count <= '0;
                end
            end
        end
    end
endmodule