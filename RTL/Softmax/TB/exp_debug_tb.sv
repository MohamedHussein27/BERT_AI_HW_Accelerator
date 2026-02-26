//============================================================================
// Debug testbench: trace 5 specific test cases through PLA exp pipeline
//============================================================================

`timescale 1ns/1ps

module exp_debug_tb;

  import softmax_pkg::*;

  parameter int CLK_PERIOD = 10;

  logic                    clk;
  logic                    rst_n;
  logic                    in_valid;
  logic signed [DATA_W-1:0] in_data;
  logic                    out_valid;
  logic [EXP_W-1:0]        out_data;

  softmax_exp_pla #(
    .D_W       (DATA_W),
    .FRAC_I    (FRAC_IN),
    .O_W       (EXP_W),
    .FRAC_O    (FRAC_EXP),
    .NSEG      (PLA_NSEG),
    .COEFF_W   (PLA_COEFF_W),
    .COEFF_F   (PLA_COEFF_F),
    .H_SHIFT   (PLA_H_SHIFT)
  ) dut (
    .clk       (clk),
    .rst_n     (rst_n),
    .in_valid  (in_valid),
    .in_data   (in_data),
    .out_valid (out_valid),
    .out_data  (out_data)
  );

  initial clk = 0;
  always #(CLK_PERIOD/2) clk = ~clk;

  // monitor internal signals
  always @(posedge clk) begin
    if (dut.s1_valid) begin
      $display("  [S1] w=%04h, x_local=%08h, w_ext=%012h, xl_ext=%012h, product=%012h, b=%04h",
               dut.s1_w, dut.s1_x_local, dut.s1_w_ext, dut.s1_xl_ext, dut.s1_product, dut.s1_b);
    end
    if (dut.s2_valid) begin
      $display("  [S2] product=%012h, scaled=%05h, b=%04h, sum=%05h",
               dut.s2_product, dut.s2_scaled, dut.s2_b, dut.s2_sum);
    end
    if (out_valid) begin
      $display("  [OUT] out_data=%04h", out_data);
    end
  end

  initial begin
    rst_n    = 0;
    in_valid = 0;
    in_data  = 0;

    repeat(5) @(posedge clk);
    rst_n = 1;
    repeat(2) @(posedge clk);

    // Test case 1: x = -1.0 in Q5.26 = 0xFC000000
    $display("\n--- Test: x = -1.0 (0xFC000000) ---");
    @(posedge clk);
    in_valid <= 1;
    in_data  <= 32'hFC000000;
    @(posedge clk);
    in_valid <= 0;
    repeat(6) @(posedge clk);

    // Test case 2: x = 0.0 in Q5.26 = 0x00000000
    $display("\n--- Test: x = 0.0 (0x00000000) ---");
    @(posedge clk);
    in_valid <= 1;
    in_data  <= 32'h00000000;
    @(posedge clk);
    in_valid <= 0;
    repeat(6) @(posedge clk);

    // Test case 3: x = -0.5 in Q5.26 = 0xFE000000
    $display("\n--- Test: x = -0.5 (0xFE000000) ---");
    @(posedge clk);
    in_valid <= 1;
    in_data  <= 32'hFE000000;
    @(posedge clk);
    in_valid <= 0;
    repeat(6) @(posedge clk);

    // Test case 4: x = -16.0 in Q5.26 = 0xC0000000
    $display("\n--- Test: x = -16.0 (0xC0000000) ---");
    @(posedge clk);
    in_valid <= 1;
    in_data  <= 32'hC0000000;
    @(posedge clk);
    in_valid <= 0;
    repeat(6) @(posedge clk);

    // Test case 5: x = -8.0 in Q5.26 = 0xE0000000
    $display("\n--- Test: x = -8.0 (0xE0000000) ---");
    @(posedge clk);
    in_valid <= 1;
    in_data  <= 32'hE0000000;
    @(posedge clk);
    in_valid <= 0;
    repeat(6) @(posedge clk);

    $finish;
  end

endmodule
