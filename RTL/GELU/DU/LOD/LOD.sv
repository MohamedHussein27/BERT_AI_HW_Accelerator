module LOD #(
  parameter int W = 32
)(
  input  logic                clk,
  input  logic                rst_n,
  input  logic                valid_in,
  input  logic [W-1:0]        data_in,
  output logic                valid_out,
  output logic [$clog2(W)-1:0] lod_pos,  // Returns actual bit position (31 to 0)
  output logic                found
);

  logic [$clog2(W)-1:0] lod_pos_comb;
  logic                  found_comb;

  always_comb begin
    found_comb   = |data_in;  // OR reduction: 1 if any bit is set
    lod_pos_comb = '0;

    if (found_comb) begin
      // Find first '1' from MSB → LSB
      // Returns the actual bit position (31 for MSB, 0 for LSB)
      for (int i = W-1; i >= 0; i--) begin
        if (data_in[i]) begin
          lod_pos_comb = i[$clog2(W)-1:0];
          break;  // Exit on first match
        end
      end
    end
  end

  // Registered outputs (1-cycle latency)
  always_ff @(posedge clk or negedge rst_n) begin
    if (!rst_n) begin
      lod_pos   <= '0;
      found     <= 1'b0;
      valid_out <= 1'b0;
    end else begin
      lod_pos   <= lod_pos_comb;
      found     <= found_comb;
      valid_out <= valid_in;
    end
  end

endmodule
