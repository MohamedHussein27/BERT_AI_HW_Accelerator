module quantize #(
    parameter DATAWIDTH_in  = 32,
    parameter DATAWIDTH_out = 8,
    parameter M_width       = 8,
    parameter S_width       = 8
)(
    input  logic                                 clk,
    input  logic                                 rst_n,
    input  logic                                 valid_in,
    input  logic signed [DATAWIDTH_in-1:0]       data_in,
    input  logic        [M_width-1:0]            scale_M,
    input  logic signed [S_width-1:0]            scale_S,

    output logic signed [DATAWIDTH_out-1:0]      data_out,
    output logic                                 valid_out
);
    localparam MUL_WIDTH = M_width + DATAWIDTH_in;
    
    // Calculate max and min values for clamping at compile-time
    localparam signed [DATAWIDTH_out-1:0] MAX_VAL = (2**(DATAWIDTH_out-1)) - 1;
    localparam signed [DATAWIDTH_out-1:0] MIN_VAL = -(2**(DATAWIDTH_out-1));
        
    // wires and regs
    logic signed [MUL_WIDTH-1:0] mul_wire;
    logic signed [MUL_WIDTH-1:0] mul_reg;
    logic                        valid_in_reg;

    // --------------------------------------------------------
    // Stage 1: Multiplication
    // --------------------------------------------------------
    assign mul_wire = data_in * $signed({1'b0, scale_M}); 

    always_ff @(posedge clk or negedge rst_n) begin : multiplication_block
        if (!rst_n) begin
            mul_reg      <= '0;
            valid_in_reg <= 1'b0;
        end else begin
            valid_in_reg <= valid_in;
            if (valid_in) begin
                mul_reg <= mul_wire;
            end  
        end
    end

    // --------------------------------------------------------
    // Stage 2: Shift, Round, and Clamp (Combinational)
    // --------------------------------------------------------
    logic signed [MUL_WIDTH-1:0]     shifted_wire;
    logic signed [DATAWIDTH_out-1:0] clamp_wire;

    assign shifted_wire = (mul_reg + (MUL_WIDTH'(1) << (30 + scale_S))) >>> (31 + scale_S);

    // Using the pre-calculated limits makes this block incredibly clean
    always_comb begin : clamp_block
        if (shifted_wire >= MAX_VAL) begin
            clamp_wire = MAX_VAL;           
        end else if (shifted_wire <= MIN_VAL) begin
            clamp_wire = MIN_VAL;          
        end else begin
            clamp_wire = shifted_wire[DATAWIDTH_out-1:0]; 
        end
    end

    // --------------------------------------------------------
    // Stage 3: Output Register
    // --------------------------------------------------------
    always_ff @(posedge clk or negedge rst_n) begin : output_block
        if (!rst_n) begin
            data_out  <= '0;
            valid_out <= 1'b0;
        end else begin
            valid_out <= valid_in_reg;
            if (valid_in_reg) begin
                data_out <= clamp_wire; 
            end
        end
    end
endmodule