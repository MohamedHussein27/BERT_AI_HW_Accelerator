`timescale 1ns/1ps

module pla_exp_tb_pipelined;

  // =========================
  // Config (match RTL)
  // =========================
  localparam int Q        = 26;    // fractional bits (Q5.26)
  localparam int W        = 32;    // word width
  localparam int NSEG     = 32;    // segments in ROM
  // Make sure NUM_TEST matches the number of lines in inputs.hex / expected.hex
  localparam int NUM_TEST = 1000;

  // tolerance for golden vs true (per-mille). e.g., 10 -> 1.0%
  localparam int REL_TOL_MILLI = 10;

  // domain constants (must match RTL's)
  localparam signed [W-1:0] XMIN = -16 << Q;
  localparam signed [W-1:0] XMAX =  16 << Q;

  // =========================
  // DUT interface signals
  // =========================
  reg                     clk;
  reg                     rst_n;
  reg                     start;
  reg signed [W-1:0]      x_in_q;
  wire                    busy;
  wire                    done;
  wire signed [W-1:0]     y_exp_q;

  // Instantiate DUT (name must match your RTL module)
  pla_exp_pipelined #(
    .Q(Q),
    .W(W),
    .H_SHIFT(Q),
    .NSEG(NSEG)
  ) uut (
    .clk(clk),
    .rst_n(rst_n),
    .start(start),
    .x_in_q(x_in_q),
    .busy(busy),
    .done(done),
    .y_exp_q(y_exp_q)
  );

  // =========================
  // Memories for vectors
  // =========================
  reg signed [W-1:0] inputs_mem  [0:NUM_TEST-1];
  reg signed [W-1:0] expect_mem  [0:NUM_TEST-1];

  // =========================
  // Test variables (module scope)
  // =========================
  integer i;
  integer timeout_cycles;
  integer fail_count;
  integer rtl_vs_gold_fail;
  integer approx_fail;
  integer too_many_words_warning;

  // stats
  integer max_rel_milli;
  real    max_abs_err_f;

  // temporaries for arithmetic/diagnostics
  reg signed [W-1:0] x_clamped_local;
  reg signed [W-1:0] delta_local;
  integer idx_ref;
  reg signed [W-1:0] w_ref_32;
  reg signed [W-1:0] b_ref_32;
  reg signed [63:0]  prod_ref_64;
  reg signed [63:0]  scaled_ref_64;
  reg signed [W-1:0] golden_q;        // golden fixed-point Q format
  reg signed [W-1:0] dut32;
  real gold_f;
  real true_f;
  real dut_f;
  real abs_err_f;
  real rel_err;
  integer rel_milli;

  // =========================
  // Clock
  // =========================
  initial begin
    clk = 1'b0;
    forever #5 clk = ~clk; // 100 MHz
  end

  // =========================
  // Load ROMs and vectors
  // =========================
  initial begin
    #1;
    $display("Loading ROMs and test vectors...");
    // load coefficient ROMs into DUT memories
    $readmemh("pla_w.hex", uut.w_mem);
    $readmemh("pla_b.hex", uut.b_mem);
    // load inputs & expected
    $readmemh("inputs.hex", inputs_mem);
    $readmemh("expected.hex", expect_mem);
    $display("ROMs and test vectors loaded.");

    // quick diagnostic print
    $display("ROM samples (w/b) [hex] and first input/expected:");
    $display(" w0=%h b0=%h  w1=%h b1=%h  w2=%h b2=%h  w3=%h b3=%h",
             uut.w_mem[0], uut.b_mem[0],
             uut.w_mem[1], uut.b_mem[1],
             uut.w_mem[2], uut.b_mem[2],
             uut.w_mem[3], uut.b_mem[3]);
    $display(" inputs[0]=%h (signed=%0d float=%f)",
             inputs_mem[0], $signed(inputs_mem[0]), $itor($signed(inputs_mem[0]))/(1<<Q));
    $display(" expected[0]=%h (signed=%0d float=%f)",
             expect_mem[0], $signed(expect_mem[0]), $itor($signed(expect_mem[0]))/(1<<Q));
  end

  // =========================
  // Main test process
  // =========================
  initial begin
    // initialize
    rst_n = 1'b0;
    start = 1'b0;
    x_in_q = '0;
    fail_count = 0;
    rtl_vs_gold_fail = 0;
    approx_fail = 0;
    max_rel_milli = 0;
    max_abs_err_f = 0.0;

    // reset pulse
    #20;
    rst_n = 1'b1;
    #20;

    // loop tests
    for (i = 0; i < NUM_TEST; i = i + 1) begin
      // apply input and pulse start (one cycle)
      @(negedge clk);
      x_in_q = inputs_mem[i];
      start  = 1'b1;
      @(negedge clk);
      start  = 1'b0;

      // wait for done with timeout
      timeout_cycles = 0;
      while (!done) begin
        @(negedge clk);
        timeout_cycles = timeout_cycles + 1;
        if (timeout_cycles > 500) begin
          $display("ERROR: Timeout waiting for done at test %0d", i);
          $finish;
        end
      end

      // sample DUT output
      dut32 = y_exp_q;

      // ------------------------
      // compute golden fixed-point PLA (same math as RTL)
      // ------------------------
      // clamp input
      if ($signed(inputs_mem[i]) < $signed(XMIN)) x_clamped_local = XMIN;
      else if ($signed(inputs_mem[i]) > $signed(XMAX)) x_clamped_local = XMAX;
      else x_clamped_local = inputs_mem[i];

      // delta & index (H_SHIFT == Q)
      delta_local = $signed(x_clamped_local) - $signed(XMIN);
      idx_ref = ($unsigned(delta_local) >>> Q);
      if (idx_ref >= NSEG) idx_ref = NSEG - 1;

      // fetch ROM coeffs from DUT
      w_ref_32 = uut.w_mem[idx_ref];
      b_ref_32 = uut.b_mem[idx_ref];

      // product and scale: scaled = (w * x) >>> Q
      prod_ref_64   = $signed(w_ref_32) * $signed(x_clamped_local); // 32x32->64
      // optional rounding: prod_ref_64 = prod_ref_64 + (64'sd1 << (Q-1));
      scaled_ref_64 = $signed(prod_ref_64) >>> Q;
      // add intercept (sign-extend b_ref_32 implicitly by assignment to 64 then truncate)
      golden_q = scaled_ref_64[W-1:0] + b_ref_32;

      // ------------------------
      // Compare DUT vs GOLDEN (exact)
      // ------------------------
      if (dut32 !== golden_q) begin
        rtl_vs_gold_fail = rtl_vs_gold_fail + 1;
        $display("RTL!=GOLDEN idx=%0d  x=%h dut=%h gold=%h  idx_ref=%0d w=%h b=%h prod=%h scaled=%h",
                 i, inputs_mem[i], dut32, golden_q, idx_ref,
                 w_ref_32, b_ref_32, prod_ref_64, scaled_ref_64);
      end

      // ------------------------
      // Compare GOLDEN vs expected (true exp) with tolerance
      // ------------------------
      gold_f = $itor($signed(golden_q)) / (1 << Q);
      true_f = $itor($signed(expect_mem[i])) / (1 << Q); // expected was produced by Python exp(x)
      dut_f  = $itor($signed(dut32)) / (1 << Q);

      // absolute and relative error
      if (gold_f >= true_f) abs_err_f = gold_f - true_f;
      else abs_err_f = true_f - gold_f;

      if (true_f == 0.0) rel_err = (abs_err_f); // fallback absolute when ref is 0
      else rel_err = abs_err_f / ( (true_f < 0.0) ? -true_f : true_f );

      // convert to per-mille (integer)
      rel_milli = $rtoi(rel_err * 1000.0);

      // update stats
      if (rel_milli > max_rel_milli) max_rel_milli = rel_milli;
      if (abs_err_f > max_abs_err_f) max_abs_err_f = abs_err_f;

      if (rel_milli > REL_TOL_MILLI) begin
        approx_fail = approx_fail + 1;
        $display("APPROX_FAIL idx=%0d rel_milli=%0d (> %0d): x_float=%f gold=%f exp_ref=%f dut=%f",
                 i, rel_milli, REL_TOL_MILLI,
                 $itor($signed(inputs_mem[i]))/(1<<Q), gold_f, true_f, dut_f);
      end

      // small settle
      @(negedge clk);
    end

    // =========================
    // Summary
    // =========================
    $display("---- Test summary ----");
    $display("Total tests: %0d", NUM_TEST);
    $display("RTL vs GOLDEN mismatches: %0d", rtl_vs_gold_fail);
    $display("GOLDEN vs true(exp) approximation fails (per-mille>%0d): %0d", REL_TOL_MILLI, approx_fail);
    $display("Max relative error (per-mille) observed: %0d", max_rel_milli);
    $display("Max absolute error (float) observed: %0.6e", max_abs_err_f);
    $display("-----------------------");
    $finish;
  end

endmodule
