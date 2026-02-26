`timescale 1ns/1ps
module tb_mul_pipelined;

  parameter AW = 32;
  parameter BW = 32;
  parameter PIPELINE_STAGES = 1;
  parameter OUTW = AW + BW;

  reg clk;
  reg rst_n;
  reg start;
  reg signed [AW-1:0] a_in;
  reg signed [BW-1:0] b_in;
  wire busy;
  wire done;
  wire signed [OUTW-1:0] p_out;

  integer i;
  integer errors;

  // golden result declared at module scope (fixes the "declarations after statements" error)
  reg signed [OUTW-1:0] golden;

  // instantiate DUT
  mul_pipelined #(
    .AW(AW),
    .BW(BW),
    .PIPELINE_STAGES(PIPELINE_STAGES)
  ) dut (
    .clk(clk),
    .rst_n(rst_n),
    .start(start),
    .a_in(a_in),
    .b_in(b_in),
    .busy(busy),
    .done(done),
    .p_out(p_out)
  );

  // clock
  initial begin
    clk = 0;
    forever #5 clk = ~clk;
  end

  // simple vector list (module-scope arrays)
  reg signed [AW-1:0] a_vec [0:9];
  reg signed [BW-1:0] b_vec [0:9];

  initial begin
    // fill vectors
    a_vec[0] = 32'sd0;         b_vec[0] = 32'sd0;
    a_vec[1] = 32'sd1;         b_vec[1] = 32'sd1;
    a_vec[2] = -32'sd1;        b_vec[2] = 32'sd2;
    a_vec[3] = 32'sd123456;    b_vec[3] = 32'sd98765;
    a_vec[4] = -32'sd123456;   b_vec[4] = -32'sd98765;
    a_vec[5] = 32'h7fffffff;   b_vec[5] = 32'sd1;       // large value
    a_vec[6] = -32'sd2147483648; b_vec[6] = 32'sd1;     // minimum signed
    a_vec[7] = -32'sd2147483648; b_vec[7] = 32'sd2;
    a_vec[8] = 32'sd1000;      b_vec[8] = -32'sd2000;
    a_vec[9] = -32'sd50000;    b_vec[9] = 32'sd40000;

    // reset
    rst_n = 0;
    start = 0;
    a_in  = 0;
    b_in  = 0;
    errors = 0;
    #100;
    rst_n = 1;
    #20;

    $display("Starting multiplication tests (PIPELINE_STAGES=%0d)", PIPELINE_STAGES);

    for (i = 0; i < 10; i = i + 1) begin
      // present operands and pulse start for one cycle
      @(negedge clk);
      a_in = a_vec[i];
      b_in = b_vec[i];
      start = 1;
      @(negedge clk);
      start = 0;

      // wait for done
      while (!done) @(negedge clk);

      // golden calculation (use signed multiply)
      golden = $signed(a_vec[i]) * $signed(b_vec[i]);

      if (p_out !== golden) begin
        $display("FAIL %0d: a=%0d b=%0d p_out=%h golden=%h", i, a_vec[i], b_vec[i], p_out, golden);
        errors = errors + 1;
      end else begin
        $display("PASS %0d: a=%0d b=%0d p_out=%h", i, a_vec[i], b_vec[i], p_out);
      end

      // small gap
      repeat (2) @(negedge clk);
    end

    $display("Tests finished. errors=%0d", errors);
    $finish;
  end

endmodule
