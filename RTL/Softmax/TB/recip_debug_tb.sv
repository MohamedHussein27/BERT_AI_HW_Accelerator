`timescale 1ns/1ps

module recip_debug_tb;
  import softmax_pkg::*;

  parameter int CLK_PERIOD = 10;

  logic                clk;
  logic                rst_n;
  logic                in_valid;
  logic [ACC_W-1:0]    in_data;
  logic                out_valid;
  logic [ACC_W-1:0]    out_data;

  softmax_reciprocal #(
    .W(ACC_W), .Q(FRAC_ACC), .LUT_BITS(NR_LUT_BITS), .ITER(NR_ITER)
  ) dut (
    .clk(clk), .rst_n(rst_n),
    .in_valid(in_valid), .in_data(in_data),
    .out_valid(out_valid), .out_data(out_data)
  );

  initial clk = 0;
  always #(CLK_PERIOD/2) clk = ~clk;

  // Monitor all internal signals every clock
  always @(posedge clk) begin
    $display("  t=%0t state=%0d a=0x%08h y=0x%08h mul_ay=0x%016h sub=0x%08h out_v=%b out_d=0x%08h",
             $time, dut.state, dut.a_norm, dut.y_reg,
             dut.mul_ay_reg, dut.sub_reg,
             dut.out_valid, dut.out_data);
  end

  initial begin
    rst_n = 0; in_valid = 0; in_data = 0;
    repeat(5) @(posedge clk);
    rst_n = 1;
    repeat(2) @(posedge clk);

    $display("\n--- Test: a = 1.0 (0x01000000) ---");
    @(posedge clk);
    in_valid <= 1;
    in_data <= 32'h01000000;
    @(posedge clk);
    in_valid <= 0;
    
    // Wait for completion
    repeat(20) @(posedge clk);

    $finish;
  end
endmodule
