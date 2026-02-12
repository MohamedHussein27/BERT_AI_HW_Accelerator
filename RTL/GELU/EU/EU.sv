module EU #(
    parameter WIDTH = 32,
    parameter Q_IN = 22,      // Input fractional bits
    parameter Q_OUT = 16      // Output fractional bits (Q48.16)
)(
    input  wire signed [WIDTH-1:0]     s_x,          // Q10.22 input
    output wire [2:0]                  segment_index,
    input  wire signed [WIDTH-1:0]     K,            // Q10.22 (32-bit LUT)
    input  wire signed [WIDTH-1:0]     B,            // Q10.22 (32-bit LUT)
    output wire signed [2*WIDTH-1:0]   exp_result    // Q48.16 output (64-bit)
);

    // Step 1: Extract integer and fractional parts from 32-bit input
    wire signed [WIDTH-Q_IN-1:0]  s_int;
    wire [Q_IN-1:0]               s_frac;
    
    assign s_int  = s_x >>> Q_IN;
    assign s_frac = s_x[Q_IN-1:0];

    // Step 2: LUT index (top 3 bits of fraction)
    assign segment_index = s_frac[Q_IN-1:Q_IN-3];

    // Step 3: Compute mantissa using 32-bit arithmetic (Q10.22)
    wire signed [2*WIDTH-1:0] K_mult_frac_full;
    wire signed [WIDTH-1:0]   K_mult_frac;
    wire signed [WIDTH-1:0]   mantissa_32;  // Q10.22 format
    
    assign K_mult_frac_full = K * $signed({{(WIDTH-Q_IN){1'b0}}, s_frac});
    assign K_mult_frac = K_mult_frac_full >>> Q_IN;
    assign mantissa_32 = K_mult_frac + B;  // Q10.22

    // Step 4: Convert mantissa from Q10.22 to Q48.16 BEFORE extension
    // Shift right by (Q_IN - Q_OUT) = (22 - 16) = 6 bits
    wire signed [WIDTH-1:0] mantissa_q4816_32;
    assign mantissa_q4816_32 = mantissa_32 >>> (Q_IN - Q_OUT);  // Now Q10.16 in 32 bits

    // Step 5: Sign-extend the Q10.16 mantissa to 64-bit
    wire signed [2*WIDTH-1:0] mantissa_64;
    assign mantissa_64 = {{WIDTH{mantissa_q4816_32[WIDTH-1]}}, mantissa_q4816_32};

    // Step 6: Barrel shift in Q48.16 format (no overflow!)
    reg signed [2*WIDTH-1:0] shifted_result;
    
    always_comb begin
        if (s_int >= 0) begin
            // Left shift (positive exponent)
            // After shift: Q(10+s_int).16 but stored in 64-bit
            shifted_result = mantissa_64 <<< s_int;
        end else begin
            // Right shift (negative exponent) - arithmetic shift
            shifted_result = mantissa_64 >>> (-s_int);
        end
    end

    // Step 7: Output is Q48.16
    assign exp_result = shifted_result;

endmodule
