`timescale 1ns/1ps

// GCU (GELU Compute Unit) - 32 Parallel GELU Accelerator
// - Input:  32 × Q10.22 (32-bit)
// - Output: 32 × Q48.16 (64-bit)
// - LUT:    64-port SharedLUT, Q10.22 32-bit coeffs (NO sign extension needed)
module GCU #(
    parameter int Q            = 22,   // Q10.22 fractional bits
    parameter int W            = 32,   // Input/LUT width (32-bit)
    parameter int NUM_GELU     = 32,
    parameter int NUM_LUT_PORTS = 64   // 32 GELUs × 2 EUs
) (
    input  wire signed [W-1:0]     x [NUM_GELU-1:0],    // 32 × Q10.22 inputs
    output wire signed [2*W-1:0]   y [NUM_GELU-1:0]     // 32 × Q48.16 outputs ✅ FIXED
);

    // =========================================================================
    // LUT Interface Signals (all 32-bit Q10.22 - no sign extension needed)
    // =========================================================================
    wire [2:0]           segment_indices  [NUM_LUT_PORTS-1:0];
    wire signed [W-1:0]  k_coeffs         [NUM_LUT_PORTS-1:0];  // ✅ 32-bit direct
    wire signed [W-1:0]  b_intercepts     [NUM_LUT_PORTS-1:0];  // ✅ 32-bit direct

    // =========================================================================
    // SharedLUT (64 ports, Q10.22 32-bit) - direct connection, no sign extend
    // =========================================================================
    SharedLUT #(
        .Q(Q),
        .W(W),
        .NUM_SEGMENTS(8),
        .NUM_PORTS(NUM_LUT_PORTS)
    ) shared_lut_inst (
        .segment_index(segment_indices),
        .k_coeff(k_coeffs),           // ✅ direct 32-bit
        .b_intercept(b_intercepts)    // ✅ direct 32-bit
    );

    // =========================================================================
    // 32 Parallel GELU Units
    // =========================================================================
    genvar i;
    generate
        for (i = 0; i < NUM_GELU; i++) begin : gelu_array
            localparam int PORT_BASE = 2 * i;

            GELU #(
                .Q(Q),               // Q10.22
                .W(W),               // 32-bit
                .LUT_PORT_BASE(PORT_BASE)
            ) gelu_inst (
                .x              (x[i]),
                .y              (y[i]),                            // ✅ 64-bit output

                .segment_index_0(segment_indices[PORT_BASE]),
                .segment_index_1(segment_indices[PORT_BASE + 1]),

                .k_coeff_0      (k_coeffs[PORT_BASE]),            // ✅ 32-bit direct
                .b_intercept_0  (b_intercepts[PORT_BASE]),
                .k_coeff_1      (k_coeffs[PORT_BASE + 1]),
                .b_intercept_1  (b_intercepts[PORT_BASE + 1])
            );
        end
    endgenerate

endmodule
