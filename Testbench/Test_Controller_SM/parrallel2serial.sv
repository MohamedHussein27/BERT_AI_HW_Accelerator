module parrallel2serial #(
    parameter VEC_SIZE = 32,
    parameter ELEMENT_DATAWIDTH = 32
) (
    input logic clk,
    input logic rst_n,
    input logic valid_in,
    input logic [VEC_SIZE-1:0][ELEMENT_DATAWIDTH-1:0] data_in,

    output logic [ELEMENT_DATAWIDTH-1:0] data_out,
    output logic busy,
    output logic valid_out
);

    localparam COUNTER_WIDTH = $clog2(VEC_SIZE);

    logic [VEC_SIZE-1:0][ELEMENT_DATAWIDTH-1:0] data_reg;
    logic [COUNTER_WIDTH-1:0] count;
    logic start;

    assign busy = start;

    always_ff @( posedge clk or negedge rst_n ) begin
        if (!rst_n)begin
            data_reg <= '0;
            valid_out <= '0;
            start <= 0;
        end
        else begin
            if (valid_in) begin
                data_reg <= data_in;
                start <= 1;              
            end
            else begin
                data_reg <= data_reg;
            end
            // output logic
            if (start) begin
                data_out <= data_reg[count];
                valid_out <= 1;
            end
            else begin
                data_out <= '0;
                valid_out <= 0;
            end
            // start signal reseting
            if (count == VEC_SIZE - 1) begin
                start <= 0;
            end

        end
    end


    always_ff @( posedge clk or negedge rst_n ) begin : counter_block
        if (!rst_n)begin
            count <= '0;
        end
        else begin
            if (start) begin
                count <= count + 1;
                if (count == VEC_SIZE - 1) begin
                    count <= '0;
                end
            end
        end
    end
    
endmodule