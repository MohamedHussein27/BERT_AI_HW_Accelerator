`timescale 1ns/1ps

module DU #(
    parameter int Q = 16,      // Fractional bits (Q48.16)
    parameter int W = 64       // Data width (64-bit)
)(
    input  logic signed [W-1:0]  F,          // Numerator (Q48.16)
    input  logic signed [W-1:0]  s_xi,       // exp(s_xi) from EU (Q48.16)
    output logic signed [W-1:0]  exponent,   // (m1+s1) - (m2+s2) in Q48.16
    output logic                 result_sign // Sign of result
);

    // Constants
    localparam logic signed [W-1:0] ONE_Q = 64'h0000000000010000;  // 1.0 in Q48.16
    localparam logic signed [W-1:0] Q_CONST = W'(signed'(Q));
    localparam logic signed [W-1:0] LARGE_NEG_EXP = -64'sd524288;  // -8.0 in Q48.16 (-8 * 2^16)

    // Detect zero
    logic F_is_zero;
    assign F_is_zero = (F == 64'h0);

    // +1
    logic signed [W-1:0] denominator;
    assign denominator = ONE_Q + s_xi;

    // Absolute and sign
    logic                F_sign, denom_sign;
    logic [W-1:0]        F_abs, denom_abs;

    assign F_sign      = F[W-1];
    assign denom_sign  = denominator[W-1];
    assign F_abs       = F_sign ? -F : F;
    assign denom_abs   = denom_sign ? -denominator : denominator;

    // LOD
    logic [$clog2(W)-1:0] F_lod_pos, denom_lod_pos;
    logic                 F_found, denom_found;

    LOD #(.W(W)) lod_F (
        .data_in(F_abs),
        .lod_pos(F_lod_pos),
        .found(F_found)
    );

    LOD #(.W(W)) lod_denom (
        .data_in(denom_abs),
        .lod_pos(denom_lod_pos),
        .found(denom_found)
    );

    // Normalize Mantissa and compute shifts
    logic signed [W-1:0] m1, m2;
    logic signed [W-1:0] s1, s2;

    // Numerator normalization
    always_comb begin
        if (F_found) begin
            // s1 = (F_lod_pos - Q) << Q
            s1 = ($signed({1'b0, {(W-$clog2(W)-1){1'b0}}, F_lod_pos}) - Q_CONST) << Q;

            // Normalize mantissa: align MSB to bit position Q (bit 16 for Q48.16)
            if (F_lod_pos > Q)
                m1 = signed'(F_abs >>> (F_lod_pos - Q));  
            else if (F_lod_pos < Q)
                m1 = signed'(F_abs <<< (Q - F_lod_pos)); 
            else
                m1 = signed'(F_abs);
        end else begin
            m1 = '0;
            s1 = '0;
        end
    end

    // Denominator normalization
    always_comb begin
        if (denom_found) begin
            // s2 = (denom_lod_pos - Q) << Q
            s2 = ($signed({1'b0, {(W-$clog2(W)-1){1'b0}}, denom_lod_pos}) - Q_CONST) << Q;

            // Normalize mantissa: align MSB to bit position Q (bit 16 for Q48.16)
            if (denom_lod_pos > Q)
                m2 = signed'(denom_abs >>> (denom_lod_pos - Q)); 
            else if (denom_lod_pos < Q)
                m2 = signed'(denom_abs <<< (Q - denom_lod_pos));  
            else
                m2 = signed'(denom_abs);
        end else begin
            m2 = '0;
            s2 = '0;
        end
    end

    // Exponent   
    logic signed [W-1:0] normal_exponent;
    assign normal_exponent = (m1 + s1) - (m2 + s2);
    
    // If F is zero, output large negative exponent → EU2 will compute 2^(-8) ≈ 0.0039
    assign exponent = F_is_zero ? LARGE_NEG_EXP : normal_exponent;
    
    // Sign handling: zero result is always positive
    assign result_sign = F_is_zero ? 1'b0 : (F_sign ^ denom_sign);

endmodule
