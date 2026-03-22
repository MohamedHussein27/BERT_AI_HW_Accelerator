module inv_sqrt_lut #(
    parameter int DATAWIDTH = 32,
    parameter LUT_ADDR = 4
) (
    input  logic [LUT_ADDR-1:0] index,       // Top 4 fractional bits of a_norm
    output logic signed [DATAWIDTH-1:0] x0   // Initial guess in Q<INT>.<FRAC>
);
    // LUT covers the normalized input range [0.5, 2.0).
    // All values are in Q5.26 format.

    always_comb begin
        case (index)
            4'h4: x0 = 32'sh05A82799; // 1.4142
            4'h5: x0 = 32'sh050F335D; // 1.2649
            4'h6: x0 = 32'sh049E5160; // 1.1547
            4'h7: x0 = 32'sh0446A3B0; // 1.0690
            4'h8: x0 = 32'sh04000000; // 1.0000
            4'h9: x0 = 32'sh03C57530; // 0.9428
            4'hA: x0 = 32'sh0393E581; // 0.8944
            4'hB: x0 = 32'sh036933A9; // 0.8528
            4'hC: x0 = 32'sh0343F07E; // 0.8165
            4'hD: x0 = 32'sh03230867; // 0.7845
            4'hE: x0 = 32'sh03055375; // 0.7559
            4'hF: x0 = 32'sh02EAABC4; // 0.7303

            default: x0 = 32'sh04000000; // 1.0
        endcase
    end

endmodule