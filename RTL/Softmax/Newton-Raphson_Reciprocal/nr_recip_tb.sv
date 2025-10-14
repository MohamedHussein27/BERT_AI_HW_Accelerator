`timescale 1ns/1ps

module nr_recip_debug_tb;

  // -------- parameters / module-scope declarations (Verilog-2001 style) --------
  parameter WIDTH = 32;
  parameter Q     = 26;

  reg clk;
  reg rst_n;
  reg start;
  reg signed [WIDTH-1:0] a_in_q;
  wire busy;
  wire done;
  wire signed [WIDTH-1:0] y_out_q;

  // test vectors array (module scope)
  integer NUM_TEST;
  initial NUM_TEST = 8; // change if you want more

  real tests_real [0:7];
  integer tests_q  [0:7];

  integer i;
  integer timeout_cnt;
  real y_ref;
  real y_dut;

  // ---------- function: convert real -> Q fixed (integer) ----------
  function integer real_to_q;
    input real r;
    begin
      real_to_q = $rtoi(r * (1 << Q));
    end
  endfunction

  // --------------- DUT instantiation ---------------
  // Adjust module name / port names if different in your RTL
  // Expected ports: clk, rst_n, start, a_in_q, busy, done, y_out_q
  nr_recip dut (
    .clk    (clk),
    .rst_n  (rst_n),
    .start  (start),
    .a_in_q (a_in_q),
    .busy   (busy),
    .done   (done),
    .y_out_q(y_out_q)
  );

  // --------------- clock ---------------
  initial begin
    clk = 0;
    forever #5 clk = ~clk; // 100 MHz
  end

  // --------------- prepare tests (module scope uses above arrays) ---------------
  initial begin
    // populate a few interesting tests (from your fail logs)
    tests_real[0] = 1.0;
    tests_real[1] = 2.0;
    tests_real[2] = 11.20603;
    tests_real[3] = 11.28593;
    tests_real[4] = 15.920101;
    tests_real[5] = 16.0;
    tests_real[6] = 0.5;
    tests_real[7] = 3.14159;

    for (i = 0; i < NUM_TEST; i = i + 1) begin
      tests_q[i] = real_to_q(tests_real[i]);
    end
  end

  // --------------- optional: load init ROM (only if your DUT exposes a memory) ---------------
  // If your DUT declares an internal memory `init_rom` you may uncomment the next lines
  // and provide a file `nr_init.hex`. If your DUT does not expose that symbol, leave commented.
  /*
  initial begin
    #1;
    $display("Attempting to load nr_init.hex into dut.init_rom (only if present)...");
    $readmemh("nr_init.hex", dut.init_rom);
    $display("Done load (if memory exists).");
  end
  */

  // --------------- test sequence ---------------
  initial begin
    // reset
    rst_n = 0;
    start = 0;
    a_in_q = 0;
    #20;
    rst_n = 1;
    #20;

    $display("Starting debug run (NUM_TEST=%0d)...", NUM_TEST);
    for (i = 0; i < NUM_TEST; i = i + 1) begin
      a_in_q = tests_q[i];
      // pulse start on falling edge to align with typical capture
      @(negedge clk);
      start = 1;
      @(negedge clk);
      start = 0;

      // wait for done, with timeout
      timeout_cnt = 0;
      while (!done) begin
        @(negedge clk);
        timeout_cnt = timeout_cnt + 1;
        if (timeout_cnt > 500) begin
          $display("TIMEOUT waiting for done for test index %0d (a_real=%f, a_q=%08h)", i, tests_real[i], a_in_q);
          $finish;
        end
      end

      // sample result
      y_dut = $itor($signed(y_out_q)) / (1 << Q);
      y_ref = 1.0 / tests_real[i];

      $display("--------------------------------------------------");
      $display("test %0d: a_real=%f  a_q=%08h", i, tests_real[i], a_in_q);
      $display(" GOLDEN reciprocal (float) = %f", y_ref);
      $display(" DUT    (hex Q%d) = %08h  (float=%f)", Q, y_out_q, y_dut);
      $display(" DUT signals: busy=%b done=%b", busy, done);
      // don't try to access dut internals here (to keep this TB generic)
      @(negedge clk);
    end

    $display("Debug run finished.");
    $finish;
  end

endmodule
