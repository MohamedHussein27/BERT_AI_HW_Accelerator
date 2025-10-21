module EU #(
    parameter int Q = 26,      // fractional bits (Q5.26)
    parameter int W = 32       // data width (bits)
) (
    input  logic                      clk,
    input  logic                      rst_n,
    input  logic                      valid_in,
    input  logic signed [W-1:0]       x,       // input in Q5.26 signed format
    output logic signed [W-1:0]       EU_out
);

    // k_fixed and b_fixed are in Q-format (Q fractional bits)
    localparam logic signed [W-1:0] k_fixed [0:7] = '{
        32'h03270154, // 0.788091
        32'h03531833, // 0.831147
        32'h03843201, // 0.879097
        32'h03B9C8C9, // 0.931430
        32'h03F38045, // 0.987794
        32'h043210BF, // 1.048892
        32'h04765343, // 1.115552
        32'h04C146E5  // 1.188747
    };

    localparam logic signed [W-1:0] b_fixed [0:7] = '{
        32'h03F85597, // 0.992514
        32'h03E69F8C, // 0.975218
        32'h03D425AF, // 0.957175
        32'h03C04BC2, // 0.937789
        32'h03AA821F, // 0.916512
        32'h039234A8, // 0.892779
        32'h0376BA8C, // 0.865946
        32'h03575E20  // 0.835320
    };

    // fractional and integer parts
    logic [Q-1 : 0]                x_frac;        // raw fractional bits (unsigned)
    logic signed [W-Q-1 : 0]       x_int;         // signed integer part (W-Q bits)
    logic [2:0]                    SEG;

    // out_frac holds the Q-format fractional result (signed)
    logic signed [W-1:0]           out_frac;

    // wide product: x_frac (Q bits) * k_fixed (W bits) -> up to W+Q bits
    logic signed [W+Q-1:0]         prod;
    int shift_amt;
    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            x_frac   <= '0;
            x_int    <= '0;
            SEG      <= '0;
            prod     <= '0;
            out_frac <= '0;
            EU_out   <= '0;
        end else begin
            if (valid_in) begin
                // extract integer & fraction parts
                x_int  <= $signed(x[W-1 : Q]);   // signed integer part
                x_frac <= x[Q-1 : 0];           // fractional bits
                SEG    <= x[Q-1 : Q-3];        // top 3 fractional bits = segment index
            end

            prod = $signed({{W{1'b0}}, x_frac}) * $signed(k_fixed[SEG]);

            out_frac <= $signed(prod >>> Q) + $signed(b_fixed[SEG]);
            shift_amt = $signed(x_int);

            if (shift_amt >= 0) begin
                if (shift_amt >= W) begin
                    if (out_frac[W-1] == 1'b0)
                        EU_out <= {1'b0, {(W-1){1'b1}}}; // saturate to large positive (approx)
                    else
                        EU_out <= {W{1'b1}};             // saturate negative (all ones)
                end else begin
                    EU_out <= $signed(out_frac <<< shift_amt);
                end
            end else begin
                shift_amt = -shift_amt;
                if (shift_amt >= W) begin
                    EU_out <= {W{out_frac[W-1]}};
                end else begin
                    EU_out <= $signed(out_frac >>> shift_amt);
                end
            end
        end
    end

endmodule
