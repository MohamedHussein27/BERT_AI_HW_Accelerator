// Leading One Detector (LOD) - Finds position of MSB '1'
module LOD #(
    parameter int W = 64  // Support 64-bit for Q48.16
)(
    input  logic [W-1:0]         data_in,
    output logic [$clog2(W)-1:0] lod_pos,   // Position of leading 1
    output logic                 found      // 1 if any bit is set
);

    always_comb begin
        found   = (data_in != '0);
        lod_pos = '0;

        if (found) begin
            // Priority encoder: find MSB position
            for (int i = W-1; i >= 0; i--) begin
                if (data_in[i]) begin
                    lod_pos = i[$clog2(W)-1:0];
                    break;  // Found MSB, exit loop
                end
            end
        end
    end

endmodule
