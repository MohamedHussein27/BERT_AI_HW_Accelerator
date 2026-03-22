`timescale 1ns/1ps
module tb_adder_tree;

    // --------------------------------------------------------
    // Format parameters
    // --------------------------------------------------------
    localparam DATAWIDTH      = 32;
    localparam FRAC_BITS      = 26;
    localparam NUM_INPUTS     = 32;
    localparam PIPELINE_DEPTH = 5;

    // Q5.26 saturation boundaries — must match adder.sv localparams exactly
    // MAX = sign=0, all integer bits 1, all frac bits 1 → 0x7FFFFFFF (+31.999999985)
    // MIN = sign=1, all integer bits 0, all frac bits 0 → 0x80000000 (-32.0)
    localparam signed [DATAWIDTH-1:0] MAX_VAL =  32'sh7FFFFFFF;  // +31.999999985
    localparam signed [DATAWIDTH-1:0] MIN_VAL = -32'sh80000000;  // -32.0

    // Named Q5.26 constants (raw = real × 2^26)
    localparam signed [DATAWIDTH-1:0] Q_0      =  32'sh00000000;  //   0.0
    localparam signed [DATAWIDTH-1:0] Q_1p5    =  32'sh06000000;  //   1.5
    localparam signed [DATAWIDTH-1:0] Q_2p5    =  32'sh0A000000;  //   2.5
    localparam signed [DATAWIDTH-1:0] Q_4p0    =  32'sh10000000;  //   4.0
    localparam signed [DATAWIDTH-1:0] Q_3p0    =  32'sh0C000000;  //   3.0
    localparam signed [DATAWIDTH-1:0] Q_3p75   =  32'sh0F000000;  //   3.75
    localparam signed [DATAWIDTH-1:0] Q_8p0    =  32'sh20000000;  //   8.0
    localparam signed [DATAWIDTH-1:0] Q_16p0   =  32'sh40000000;  //  16.0
    localparam signed [DATAWIDTH-1:0] Q_17p0   =  32'sh44000000;  //  17.0
    localparam signed [DATAWIDTH-1:0] Q_0p25   =  32'sh01000000;  //   0.25
    localparam signed [DATAWIDTH-1:0] Q_0p5    =  32'sh02000000;  //   0.5
    localparam signed [DATAWIDTH-1:0] Q_1LSB   =  32'sh00000001;  //   2^-26 (smallest step)
    localparam signed [DATAWIDTH-1:0] Q_N1p5   = -32'sh06000000;  //  -1.5
    localparam signed [DATAWIDTH-1:0] Q_N2p5   = -32'sh0A000000;  //  -2.5
    localparam signed [DATAWIDTH-1:0] Q_N4p0   = -32'sh10000000;  //  -4.0
    localparam signed [DATAWIDTH-1:0] Q_N3p75  = -32'sh0F000000;  //  -3.75
    localparam signed [DATAWIDTH-1:0] Q_N8p0   = -32'sh20000000;  //  -8.0
    localparam signed [DATAWIDTH-1:0] Q_N16p0  = -32'sh40000000;  // -16.0
    localparam signed [DATAWIDTH-1:0] Q_N17p0  = -32'sh44000000;  // -17.0
    localparam signed [DATAWIDTH-1:0] Q_N0p25  = -32'sh01000000;  //  -0.25
    localparam signed [DATAWIDTH-1:0] Q_N0p5   = -32'sh02000000;  //  -0.5
    localparam signed [DATAWIDTH-1:0] Q_N1LSB  = -32'sh00000001;  //  -2^-26

    // --------------------------------------------------------
    // Shared signals
    // --------------------------------------------------------
    logic clk;
    logic rst_n;
    logic valid_in;

    // --------------------------------------------------------
    // Adder unit-test signals
    // --------------------------------------------------------
    logic signed [DATAWIDTH-1:0] a_in1, a_in2, a_out;

    // --------------------------------------------------------
    // Adder-tree signals
    // --------------------------------------------------------
    logic signed [DATAWIDTH-1:0] data_in [NUM_INPUTS-1:0];
    logic signed [DATAWIDTH-1:0] result;
    logic                        tree_valid_out;

    // --------------------------------------------------------
    // Clock: 10 ns period (100 MHz)
    // --------------------------------------------------------
    initial clk = 1'b0;
    always #5 clk = ~clk;

    // --------------------------------------------------------
    // DUT 1 — adder
    // --------------------------------------------------------
    adder #(
        .DATAWIDTH(DATAWIDTH),
        .FRAC_BITS(FRAC_BITS)
    ) u_adder (
        .clk      (clk),
        .rst_n    (rst_n),
        .valid_in (valid_in),
        .data_in_1(a_in1),
        .data_in_2(a_in2),
        .data_out (a_out)
    );

    // --------------------------------------------------------
    // DUT 2 — adder_tree
    // --------------------------------------------------------
    adder_tree #(
        .DATAWIDTH    (DATAWIDTH),
        .FRAC_BITS    (FRAC_BITS),
        .NUM_OF_INPUTS(NUM_INPUTS)
    ) u_tree (
        .clk      (clk),
        .rst_n    (rst_n),
        .valid_in (valid_in),
        .data_in  (data_in),
        .result   (result),
        .valid_out(tree_valid_out)
    );

    // --------------------------------------------------------
    // Pass / fail counters
    // --------------------------------------------------------
    int pass_count = 0;
    int fail_count = 0;

    // --------------------------------------------------------
    // Helper: convert raw Q5.26 register value to real
    // --------------------------------------------------------
    function automatic real to_real (input signed [DATAWIDTH-1:0] v);
        return real'($signed(v)) / real'(2**FRAC_BITS);
    endfunction

    // --------------------------------------------------------
    // Task: adder unit test
    // --------------------------------------------------------
    task automatic test_adder (
        input signed [DATAWIDTH-1:0] in1,
        input signed [DATAWIDTH-1:0] in2,
        input signed [DATAWIDTH-1:0] expected,
        input string                 label
    );
        @(negedge clk);
        a_in1    = in1;
        a_in2    = in2;
        valid_in = 1'b1;

        @(posedge clk); #1;

        if ($signed(a_out) !== $signed(expected)) begin
            $display("FAIL [%-28s]  got=%14.8f  expected=%14.8f  (raw: got=%0d exp=%0d)",
                     label, to_real(a_out), to_real(expected),
                     $signed(a_out), $signed(expected));
            fail_count++;
        end else begin
            $display("PASS [%-28s]  %10.6f + %10.6f = %10.6f",
                     label, to_real(in1), to_real(in2), to_real(a_out));
            pass_count++;
        end

        @(negedge clk); valid_in = 1'b0;
        @(posedge clk); #1;
    endtask

    // --------------------------------------------------------
    // Task: verify data_out holds when valid_in=0
    // --------------------------------------------------------
    task automatic test_adder_hold (
        input signed [DATAWIDTH-1:0] in1,
        input signed [DATAWIDTH-1:0] in2,
        input string                 label
    );
        logic signed [DATAWIDTH-1:0] captured;

        @(negedge clk);
        a_in1 = in1; a_in2 = in2; valid_in = 1'b1;
        @(posedge clk); #1;
        captured = a_out;

        @(negedge clk); valid_in = 1'b0;
        @(posedge clk); #1;

        if ($signed(a_out) !== $signed(captured)) begin
            $display("FAIL [%-28s]  hold broken: was=%10.6f  now=%10.6f",
                     label, to_real(captured), to_real(a_out));
            fail_count++;
        end else begin
            $display("PASS [%-28s]  output held at %10.6f",
                     label, to_real(a_out));
            pass_count++;
        end

        @(negedge clk); valid_in = 1'b0;
        @(posedge clk); #1;
    endtask

    // --------------------------------------------------------
    // Task: adder_tree integration test
    // --------------------------------------------------------
    task automatic test_tree (
        input signed [DATAWIDTH-1:0] val,
        input signed [DATAWIDTH-1:0] expected,
        input string                 label
    );
        @(negedge clk);
        for (int j = 0; j < NUM_INPUTS; j++) data_in[j] = val;
        valid_in = 1'b1;

        repeat (PIPELINE_DEPTH) @(posedge clk);
        #1;

        if (tree_valid_out !== 1'b1) begin
            $display("FAIL [%-28s]  valid_out not asserted after %0d cycles",
                     label, PIPELINE_DEPTH);
            fail_count++;
        end else if ($signed(result) !== $signed(expected)) begin
            $display("FAIL [%-28s]  got=%14.8f  expected=%14.8f  (raw: got=%0d exp=%0d)",
                     label, to_real(result), to_real(expected),
                     $signed(result), $signed(expected));
            fail_count++;
        end else begin
            $display("PASS [%-28s]  32 x %10.6f = %10.6f",
                     label, to_real(val), to_real(result));
            pass_count++;
        end

        @(negedge clk); valid_in = 1'b0;
        @(posedge clk); #1;
    endtask

    // --------------------------------------------------------
    // Main stimulus
    // --------------------------------------------------------
    initial begin
        $dumpfile("tb_adder_tree.vcd");
        $dumpvars(0, tb_adder_tree);

        // ---------- initialise ----------
        rst_n    = 1'b0;
        valid_in = 1'b0;
        a_in1    = Q_0;
        a_in2    = Q_0;
        for (int k = 0; k < NUM_INPUTS; k++) data_in[k] = Q_0;

        // ---------- reset ----------
        repeat (4) @(posedge clk);
        @(negedge clk); rst_n = 1'b1;
        @(posedge clk); #1;

        $display("\n--- Reset check ---");
        if (a_out !== Q_0) begin
            $display("FAIL [reset_adder]               data_out not cleared");
            fail_count++;
        end else begin
            $display("PASS [reset_adder]               data_out=0 after reset");
            pass_count++;
        end
        if (tree_valid_out !== 1'b0) begin
            $display("FAIL [reset_tree]                valid_out not cleared");
            fail_count++;
        end else begin
            $display("PASS [reset_tree]                valid_out=0 after reset");
            pass_count++;
        end

        // =====================================================
        // SECTION 1 — Adder unit tests
        // =====================================================
        $display("\n--- Adder unit tests (Q5.26: DATAWIDTH=%0d, FRAC_BITS=%0d) ---",
                 DATAWIDTH, FRAC_BITS);
        $display("    MAX_VAL = %10.6f (0x%08X)   MIN_VAL = %10.6f (0x%08X)",
                 to_real(MAX_VAL), MAX_VAL, to_real(MIN_VAL), MIN_VAL);

        // Normal — no overflow
        test_adder(Q_1p5,    Q_2p5,    Q_4p0,              "1.5+2.5=4.0"           );
        test_adder(Q_N1p5,   Q_N2p5,   Q_N4p0,             "-1.5+-2.5=-4.0"        );
        test_adder(Q_3p0,    Q_N1p5,   Q_1p5,              "3.0+-1.5=1.5"          );
        // Zero
        test_adder(Q_0,      Q_0,      Q_0,                "0.0+0.0=0.0"           );
        // Boundary: 0 + MAX stays MAX (no overflow)
        test_adder(Q_0,      MAX_VAL,  MAX_VAL,            "0.0+MAX=MAX_noflow"    );
        // Boundary: 0 + MIN stays MIN (no overflow)
        test_adder(Q_0,      MIN_VAL,  MIN_VAL,            "0.0+MIN=MIN_noflow"    );

        // Positive saturation → MAX_VAL (0x7FFFFFFF)
        // 16.0+17.0=33.0 > 31.999... → saturate
        test_adder(Q_16p0,   Q_17p0,   MAX_VAL,            "16.0+17.0->SAT_MAX"    );
        // MAX + 1 LSB → saturate (smallest possible overflow)
        test_adder(MAX_VAL,  Q_1LSB,   MAX_VAL,            "MAX+1LSB->SAT_MAX"     );
        // MAX + MAX → saturate
        test_adder(MAX_VAL,  MAX_VAL,  MAX_VAL,            "MAX+MAX->SAT_MAX"      );

        // Negative saturation → MIN_VAL (0x80000000)
        // -16.0+-17.0=-33.0 < -32.0 → saturate
        test_adder(Q_N16p0,  Q_N17p0,  MIN_VAL,            "-16.0+-17.0->SAT_MIN"  );
        // MIN - 1 LSB → saturate (smallest possible underflow)
        test_adder(MIN_VAL,  Q_N1LSB,  MIN_VAL,            "MIN+-1LSB->SAT_MIN"    );
        // MIN + MIN → saturate
        test_adder(MIN_VAL,  MIN_VAL,  MIN_VAL,            "MIN+MIN->SAT_MIN"      );

        // Mixed signs — can never overflow regardless of magnitude
        // MAX + MIN = 31.999... + (-32.0) = -1 LSB
        test_adder(MAX_VAL,  MIN_VAL,  MAX_VAL+MIN_VAL,    "MAX+MIN_noflow=-1LSB"  );
        // 3.75 + (-3.75) = 0
        test_adder(Q_3p75,   Q_N3p75,  Q_0,                "3.75+-3.75=0.0"        );

        // =====================================================
        // SECTION 2 — valid_in=0 hold
        // =====================================================
        $display("\n--- valid_in=0 hold behaviour ---");
        test_adder_hold(Q_1p5, Q_2p5, "hold:1.5+2.5(=4.0)");

        // =====================================================
        // SECTION 3 — Mid-stream reset on adder
        // =====================================================
        $display("\n--- Mid-stream reset ---");
        @(negedge clk);
        a_in1    = Q_3p75;
        a_in2    = Q_3p75;
        valid_in = 1'b1;
        @(posedge clk);

        @(negedge clk); rst_n = 1'b0;
        @(posedge clk); #1;

        if (a_out !== Q_0) begin
            $display("FAIL [mid_reset_adder]           data_out=%10.6f, expected 0",
                     to_real(a_out));
            fail_count++;
        end else begin
            $display("PASS [mid_reset_adder]           data_out cleared by reset");
            pass_count++;
        end

        @(negedge clk); rst_n = 1'b1; valid_in = 1'b0;
        repeat (2) @(posedge clk);

        // =====================================================
        // SECTION 4 — Adder-tree integration tests
        // =====================================================
        $display("\n--- Adder tree tests (32 inputs, Q5.26) ---");

        // 32 × 0.0 = 0.0
        test_tree(Q_0,      Q_0,      "32x0.0=0.0"             );

        // 32 × 0.25 = 8.0 — fits in range (8.0 < 31.999...)
        // Stage sums: 0.5 → 1.0 → 2.0 → 4.0 → 8.0, no saturation
        test_tree(Q_0p25,   Q_8p0,    "32x0.25=8.0_noflow"     );

        // 32 × -0.25 = -8.0 — fits in range (-8.0 > -32.0)
        test_tree(Q_N0p25, -Q_8p0,    "32x-0.25=-8.0_noflow"   );

        // 32 × 1.0 = 32.0 — just exceeds MAX (32.0 > 31.999...)
        // Stage 5: 16.0+16.0=32.0 → saturates to MAX
        test_tree(Q_1p5,    MAX_VAL,  "32x1.5->SAT_MAX"        );

        // 32 × -0.5 = -16.0 — fits in range (-16.0 > -32.0), no saturation
        test_tree(Q_N0p5,   Q_N16p0,  "32x-0.5=-16.0_noflow"   );

        // 32 × 1.5 = 48.0 — saturates deep in pipeline
        // st1: 3.0, st2: 6.0, st3: 12.0, st4: 24.0, st5: 48.0→SAT
        test_tree(Q_1p5,    MAX_VAL,  "32x1.5->SAT_MAX"        );

        // 32 × -1.5 = -48.0 — saturates deep in pipeline
        test_tree(Q_N1p5,   MIN_VAL,  "32x-1.5->SAT_MIN"       );

        // =====================================================
        // SECTION 5 — valid_out flush
        // =====================================================
        $display("\n--- valid_out flush ---");
        @(negedge clk);
        for (int j = 0; j < NUM_INPUTS; j++) data_in[j] = Q_1p5;
        valid_in = 1'b1;
        repeat (PIPELINE_DEPTH) @(posedge clk);

        @(negedge clk); valid_in = 1'b0;
        repeat (PIPELINE_DEPTH + 2) @(posedge clk); #1;

        if (tree_valid_out !== 1'b0) begin
            $display("FAIL [valid_flush]               valid_out still high after drain");
            fail_count++;
        end else begin
            $display("PASS [valid_flush]               valid_out correctly deasserted");
            pass_count++;
        end

        // =====================================================
        // SECTION 6 — Back-to-back transactions
        // Burst A: 32 × 0.25 = 8.0  (no overflow)
        // Burst B: 32 × -0.25 = -8.0 (no overflow)
        // =====================================================
        $display("\n--- Back-to-back transactions ---");
        begin
            logic signed [DATAWIDTH-1:0] res_a, res_b;

            @(negedge clk);
            for (int j = 0; j < NUM_INPUTS; j++) data_in[j] = Q_0p25;
            valid_in = 1'b1;
            @(posedge clk);

            @(negedge clk);
            for (int j = 0; j < NUM_INPUTS; j++) data_in[j] = Q_N0p25;

            repeat (PIPELINE_DEPTH - 1) @(posedge clk); #1;
            res_a = result;

            @(posedge clk); #1;
            res_b = result;

            @(negedge clk); valid_in = 1'b0;

            if ($signed(res_a) !== $signed(Q_8p0)) begin
                $display("FAIL [b2b_burst_A]               got=%10.6f  expected=8.0",
                         to_real(res_a));
                fail_count++;
            end else begin
                $display("PASS [b2b_burst_A]               result=%10.6f", to_real(res_a));
                pass_count++;
            end

            if ($signed(res_b) !== $signed(-Q_8p0)) begin
                $display("FAIL [b2b_burst_B]               got=%10.6f  expected=-8.0",
                         to_real(res_b));
                fail_count++;
            end else begin
                $display("PASS [b2b_burst_B]               result=%10.6f", to_real(res_b));
                pass_count++;
            end
        end

        // =====================================================
        // SECTION 7 — Mid-stream reset on tree
        // =====================================================
        $display("\n--- Mid-stream reset on tree ---");
        @(negedge clk);
        for (int j = 0; j < NUM_INPUTS; j++) data_in[j] = Q_3p75;
        valid_in = 1'b1;
        repeat (2) @(posedge clk);

        @(negedge clk); rst_n = 1'b0;
        @(posedge clk); #1;

        if (tree_valid_out !== 1'b0) begin
            $display("FAIL [mid_reset_tree]            valid_out not cleared");
            fail_count++;
        end else begin
            $display("PASS [mid_reset_tree]            valid_out cleared by reset");
            pass_count++;
        end

        @(negedge clk); rst_n = 1'b1; valid_in = 1'b0;
        repeat (2) @(posedge clk);

        // =====================================================
        // Summary
        // =====================================================
        $display("\n========================================");
        $display("  Results:  %0d PASSED  |  %0d FAILED",
                 pass_count, fail_count);
        $display("========================================");
        if (fail_count == 0)
            $display("  ALL TESTS PASSED\n");
        else
            $display("  *** %0d TEST(S) FAILED ***\n", fail_count);

        $finish;
    end
endmodule