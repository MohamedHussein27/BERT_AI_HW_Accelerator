`timescale 1ns/1ps

module LOD_tb;
  // Parameters
  localparam int W = 32;
  localparam int NUM_TESTS = 200;

  // DUT I/O
  logic                clk;
  logic                rst_n;
  logic                valid_in;
  logic [W-1:0]        data_in;
  logic                valid_out;
  logic [$clog2(W)-1:0] lod_pos;
  logic                found;

  // DUT Instance
  LOD #(.W(W)) dut (
    .clk(clk),
    .rst_n(rst_n),
    .valid_in(valid_in),
    .data_in(data_in),
    .valid_out(valid_out),
    .lod_pos(lod_pos),
    .found(found)
  );

  // Clock generation
  always #5 clk = ~clk;

  // Reference Model
  function automatic int ref_lod_pos(input logic [W-1:0] din);
    for (int i = W-1; i >= 0; i--) begin
      if (din[i])
        return (i); // position from LSB side
    end
    return 0;
  endfunction

  function automatic bit ref_found(input logic [W-1:0] din);
    return (|din);
  endfunction

  // Test variables
  int pass_count = 0;
  int fail_count = 0;
  int ref_pos;
  bit ref_fnd;

  // Test stimulus
  initial begin
    clk = 0;
    rst_n = 0;
    valid_in = 0;
    data_in = '0;

    // Reset
    repeat(3) @(posedge clk);
    rst_n = 1;
    @(posedge clk);

    // Randomized Tests
    for (int t = 0; t < NUM_TESTS; t++) begin
      data_in = $urandom();
      valid_in = 1;
      @(posedge clk);

      ref_fnd = ref_found(data_in);
      ref_pos = ref_lod_pos(data_in);

      @(posedge clk); // wait for pipeline output

      if (valid_out) begin
        if (found === ref_fnd && lod_pos === ref_pos) begin
          pass_count++;
        end else begin
          $display("❌ [FAIL] Test %0d | data_in=%b | found=%b (exp %b) | lod_pos=%0d (exp %0d)",
                   t, data_in, found, ref_fnd, lod_pos, ref_pos);
          fail_count++;
        end
      end
    end

    // Edge Cases
    data_in = 0;
    valid_in = 1;
    @(posedge clk);
    ref_fnd = 0;
    ref_pos = 0;
    @(posedge clk);
    if (found === ref_fnd && lod_pos === ref_pos) pass_count++; else fail_count++;

    data_in = 32'h80000000;
    valid_in = 1;
    @(posedge clk);
    ref_fnd = 1;
    ref_pos = 31;
    @(posedge clk);
    if (found === ref_fnd && lod_pos === ref_pos) pass_count++; else fail_count++;

    data_in = 32'h00000001;
    valid_in = 1;
    @(posedge clk);
    ref_fnd = 1;
    ref_pos = 0;
    @(posedge clk);
    if (found === ref_fnd && lod_pos === ref_pos) pass_count++; else fail_count++;

    // Final Summary
    $display("\n========================================");
    $display(" LOD Test Summary:");
    $display("  Passed: %0d", pass_count);
    $display("  Failed: %0d", fail_count);
    $display("========================================");

    if (fail_count == 0)
      $display("✅ All tests passed!");
    else
      $display("❌ Some tests failed.");

    $finish;
  end

endmodule
