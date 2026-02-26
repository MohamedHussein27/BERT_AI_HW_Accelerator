module PolynomialUnit #(
    parameter int WIDTH = 32,           
    parameter int Q = 22                // Q10.22 format
)(
    input  wire signed [WIDTH-1:0] x,      // Q10.22 input
    output wire signed [WIDTH-1:0] s_x     // Q10.22 output
);

    // Stage 1: Compute x²
    // Input: Q10.22 × Q10.22 → Q20.44 (64-bit intermediate)
    wire signed [2*WIDTH-1:0] x_squared_full;
    wire signed [WIDTH-1:0] x_squared;
    
    assign x_squared_full = x * x;              // 32×32 = 64 bits (Q20.44)
    assign x_squared = x_squared_full >>> Q;    // Shift by 22 → Q10.22

    // Stage 2: Compute x³ = x² × x
    // Input: Q10.22 × Q10.22 → Q20.44 (64-bit intermediate)
    wire signed [2*WIDTH-1:0] x_cubed_full;
    wire signed [WIDTH-1:0] x_cubed;
    
    assign x_cubed_full = x_squared * x;        // Q10.22 × Q10.22 = Q20.44
    assign x_cubed = x_cubed_full >>> Q;        // Shift by 22 → Q10.22

    // Stage 3: Compute K2 × x³ (where K2 = 0.046875 = 3/64)
    // Method: K2 × x³ = x³/32 + x³/64 (exact representation for K2 = 0.046875)
    wire signed [WIDTH-1:0] K2_x3;
    
    assign K2_x3 = (x_cubed >>> 5) + (x_cubed >>> 6);  // K2 = 1/32 + 1/64

    // Stage 4: Compute inner = x + K2×x³
    wire signed [WIDTH-1:0] inner;
    
    assign inner = x + K2_x3;

    // Stage 5: Compute K1 × inner (where K1 = 2.3125 = 2 + 1/4 + 1/16)
    wire signed [WIDTH-1:0] K1_inner;
    
    assign K1_inner = (inner <<< 1) + (inner >>> 2) + (inner >>> 4);

    // Stage 6: Final result = -K1 × inner
    assign s_x = -K1_inner;

endmodule
