module PolynomialUnit #(
    parameter int WIDTH = 64,           
    parameter int Q = 16                // Q48.16 format
)(
    input  wire signed [WIDTH-1:0] x,      // Q48.16 input
    output wire signed [WIDTH-1:0] s_x     // Q48.16 output
);

    // Stage 1: Compute x² 
    // Input: Q48.16 × Q48.16 → Q96.32 (128-bit)
    wire signed [2*WIDTH-1:0] x_squared_full;
    wire signed [WIDTH-1:0] x_squared;
    
    assign x_squared_full = x * x;              // 64×64 = 128 bits
    assign x_squared = x_squared_full >>> Q;    // Shift by 16 → Q48.16

    // Stage 2: Compute x³ = x² × x (Q48.16)
    wire signed [2*WIDTH-1:0] x_cubed_full;
    wire signed [WIDTH-1:0] x_cubed;
    
    assign x_cubed_full = x_squared * x;        // Q96.32
    assign x_cubed = x_cubed_full >>> Q;        // Shift by 16 → Q48.16

    // Stage 3: Compute K2 × x³ (where K2 = 0.046875 = 3/64)
    // Method: K2 × x³ = x³/32 + x³/64 (exact for K2 = 0.046875)
    wire signed [WIDTH-1:0] K2_x3;
    
    assign K2_x3 = (x_cubed >>> 5) + (x_cubed >>> 6);  // K2 = 1/32 + 1/64

    // Stage 4: Compute inner = x + K2×x³ (Q48.16)
    wire signed [WIDTH-1:0] inner;
    
    assign inner = x + K2_x3;

    // Stage 5: Compute K1 × inner (where K1 = 2.3125 = 2 + 1/4 + 1/16)
    // Method: K1 × inner = 2×inner + inner/4 + inner/16
    wire signed [WIDTH-1:0] K1_inner;
    
    assign K1_inner = (inner <<< 1) + (inner >>> 2) + (inner >>> 4);

    // Stage 6: Final result = -K1 × inner (Q48.16)
    assign s_x = -K1_inner;

endmodule
