`timescale 1ns/1ps

module GCU #(
    parameter int Q = 16,              // Fractional bits (Q48.16)
    parameter int W = 64,              // Data width (64-bit)
    parameter int NUM_GELU = 32,       // Number of parallel GELU units
    parameter int NUM_LUT_PORTS = 64   // 32 GELUs × 2 EUs each
) (
    // Data Path - 32 Parallel Inputs/Outputs
    input  wire signed [W-1:0] x [NUM_GELU-1:0],    // 32 input values (Q48.16)
    output wire signed [W-1:0] y [NUM_GELU-1:0]     // 32 output values (Q48.16)
);

    // LUT Interface Signals
    // Segment indices from all GELU units to SharedLUT
    wire [2:0] segment_indices [NUM_LUT_PORTS-1:0];
    
    // Coefficients from SharedLUT to all GELU units
    // Note: SharedLUT outputs 32-bit values (Q10.22)
    wire signed [31:0] k_coeffs_32 [NUM_LUT_PORTS-1:0];
    wire signed [31:0] b_intercepts_32 [NUM_LUT_PORTS-1:0];
    
    // Sign-extended to 64-bit for GELU interface (Q48.16)
    wire signed [W-1:0] k_coeffs [NUM_LUT_PORTS-1:0];
    wire signed [W-1:0] b_intercepts [NUM_LUT_PORTS-1:0];

    // Shared LUT Instance (64 ports, Q10.22 format)
    SharedLUT #(
        .Q(22),                    // LUT uses Q10.22 format
        .W(32),                    // LUT uses 32-bit width
        .NUM_SEGMENTS(8),
        .NUM_PORTS(NUM_LUT_PORTS)
    ) shared_lut_inst (
        .segment_index(segment_indices),
        .k_coeff(k_coeffs_32),
        .b_intercept(b_intercepts_32)
    );

    // Sign-Extend LUT Outputs from 32-bit to 64-bit
    genvar p;
    generate
        for (p = 0; p < NUM_LUT_PORTS; p++) begin : lut_sign_extend
            assign k_coeffs[p] = {{32{k_coeffs_32[p][31]}}, k_coeffs_32[p]};
            assign b_intercepts[p] = {{32{b_intercepts_32[p][31]}}, b_intercepts_32[p]};
        end
    endgenerate

    // Generate 32 Parallel GELU Units
    genvar i;
    generate
        for (i = 0; i < NUM_GELU; i++) begin : gelu_array
            // Each GELU uses 2 consecutive LUT ports: 2*i and 2*i+1
            localparam int PORT_BASE = 2 * i;
            
            GELU #(
                .Q(Q),                  // Q48.16
                .W(W),                  // 64-bit
                .LUT_PORT_BASE(PORT_BASE)
            ) gelu_inst (
                // Data path (Q48.16)
                .x(x[i]),
                .y(y[i]),
                
                // LUT interface - segment indices (to LUT)
                .segment_index_0(segment_indices[PORT_BASE]),       // EU1 -> LUT port 2*i
                .segment_index_1(segment_indices[PORT_BASE + 1]),   // EU2 -> LUT port 2*i+1
                
                // LUT interface - coefficients (from LUT, 64-bit Q48.16)
                .k_coeff_0(k_coeffs[PORT_BASE]),                   // LUT port 2*i -> EU1
                .b_intercept_0(b_intercepts[PORT_BASE]),
                .k_coeff_1(k_coeffs[PORT_BASE + 1]),               // LUT port 2*i+1 -> EU2
                .b_intercept_1(b_intercepts[PORT_BASE + 1])
            );
        end
    endgenerate
endmodule
