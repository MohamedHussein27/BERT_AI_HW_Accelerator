`timescale 1ns/1ps

module ExpLUT_tb;

    // Parameters
    localparam int Q = 26;
    localparam int W = 32;
    localparam int NUM_SEGMENTS = 8;
    localparam int NUM_PORTS = 32;

    // DUT signals
    logic [2:0]              segment_index [NUM_PORTS-1:0];
    logic signed [W-1:0]     k_coeff [NUM_PORTS-1:0];
    logic signed [W-1:0]     b_intercept [NUM_PORTS-1:0];

    // Expected values for verification
    localparam logic signed [W-1:0] k_expected [0:7] = '{
        32'h02E57078, // 0.724062
        32'h03288B9B, // 0.789595
        32'h0371B996, // 0.861060
        32'h03C18722, // 0.938992
        32'h04188DB7, // 1.023978
        32'h047774AE, // 1.116656
        32'h04DEF287, // 1.217722
        32'h054FCE46  // 1.327935
    };

    localparam logic signed [W-1:0] b_expected [0:7] = '{
        32'h04000000, // 1.000000
        32'h03F79C9B, // 0.991808
        32'h03E5511D, // 0.973942
        32'h03C76408, // 0.944718
        32'h039BE0BD, // 0.902224
        32'h03609063, // 0.844301
        32'h0312F200, // 0.768501
        32'h02B031B9  // 0.672065
    };

    // Instantiate DUT
    ExpLUT #(
        .Q(Q),
        .W(W),
        .NUM_SEGMENTS(NUM_SEGMENTS),
        .NUM_PORTS(NUM_PORTS)
    ) dut (
        .segment_index(segment_index),
        .k_coeff(k_coeff),
        .b_intercept(b_intercept)
    );

    // Test variables
    int errors;
    real k_real, b_real;

    initial begin
        errors = 0;

        $display("\n========================================================================");
        $display("                    ExpLUT Module Test");
        $display("========================================================================");
        $display("Testing %0d parallel ports with %0d segments\n", NUM_PORTS, NUM_SEGMENTS);

        // Test 1: All ports access segment 0
        $display("Test 1: All ports access segment 0");
        for (int i = 0; i < NUM_PORTS; i++) begin
            segment_index[i] = 3'd0;
        end
        #10;
        
        for (int i = 0; i < NUM_PORTS; i++) begin
            if (k_coeff[i] !== k_expected[0] || b_intercept[i] !== b_expected[0]) begin
                $display("ERROR: Port %0d - Expected k=0x%08h, b=0x%08h | Got k=0x%08h, b=0x%08h",
                         i, k_expected[0], b_expected[0], k_coeff[i], b_intercept[i]);
                errors++;
            end
        end
        if (errors == 0) $display("  PASS: All ports correctly accessed segment 0\n");

        // Test 2: Each port accesses different segment (cycling through 0-7)
        $display("Test 2: Ports access different segments (cycling pattern)");
        for (int i = 0; i < NUM_PORTS; i++) begin
            segment_index[i] = i % 8;  // Cycle through segments 0-7
        end
        #10;
        
        for (int i = 0; i < NUM_PORTS; i++) begin
            automatic int expected_seg;  // Declare as automatic
            expected_seg = i % 8;
            if (k_coeff[i] !== k_expected[expected_seg] || b_intercept[i] !== b_expected[expected_seg]) begin
                $display("ERROR: Port %0d (seg %0d) - Expected k=0x%08h, b=0x%08h | Got k=0x%08h, b=0x%08h",
                         i, expected_seg, k_expected[expected_seg], b_expected[expected_seg], 
                         k_coeff[i], b_intercept[i]);
                errors++;
            end
        end
        if (errors == 0) $display("  PASS: All ports correctly accessed their assigned segments\n");

        // Test 3: All segments accessed by port 0 (verify all LUT entries)
        $display("Test 3: Verify all 8 LUT entries using port 0");
        $display("Seg |     k (hex)      k (real)   |     b (hex)      b (real)");
        $display("----|------------------------------|------------------------------");
        
        for (int seg = 0; seg < NUM_SEGMENTS; seg++) begin
            segment_index[0] = seg;
            #10;
            
            k_real = $itor(k_coeff[0]) / real'(1 << Q);
            b_real = $itor(b_intercept[0]) / real'(1 << Q);
            
            $display(" %0d  | 0x%08h  %9.6f  | 0x%08h  %9.6f", 
                     seg, k_coeff[0], k_real, b_intercept[0], b_real);
            
            if (k_coeff[0] !== k_expected[seg] || b_intercept[0] !== b_expected[seg]) begin
                $display("ERROR: Segment %0d mismatch!", seg);
                errors++;
            end
        end
        if (errors == 0) $display("\n  PASS: All LUT entries correct\n");

        // Test 4: Random access pattern
        $display("Test 4: Random access pattern");
        for (int i = 0; i < NUM_PORTS; i++) begin
            segment_index[i] = $urandom_range(0, 7);
        end
        #10;
        
        $display("Port | Seg | k_coeff (hex) | b_intercept (hex) | Status");
        $display("-----|-----|---------------|-------------------|--------");
        for (int i = 0; i < 8; i++) begin  // Show first 8 ports
            automatic int seg;
            automatic string status;
            seg = segment_index[i];
            status = (k_coeff[i] === k_expected[seg] && 
                     b_intercept[i] === b_expected[seg]) ? "PASS" : "FAIL";
            $display("  %2d |  %0d  | 0x%08h    | 0x%08h        | %s", 
                     i, seg, k_coeff[i], b_intercept[i], status);
            
            if (status == "FAIL") errors++;
        end
        $display("  ... (remaining ports also checked)");
        
        for (int i = 8; i < NUM_PORTS; i++) begin
            automatic int seg;
            seg = segment_index[i];
            if (k_coeff[i] !== k_expected[seg] || b_intercept[i] !== b_expected[seg]) begin
                errors++;
            end
        end
        if (errors == 0) $display("\n  PASS: Random access pattern correct\n");

        // Test 5: Timing check - simultaneous access
        $display("Test 5: Simultaneous multi-port access timing");
        for (int i = 0; i < NUM_PORTS; i++) begin
            segment_index[i] = (i / 4) % 8;  // Groups of 4 access same segment
        end
        
        #1;  // Check propagation delay
        $display("  After 1ns, all outputs should be stable");
        
        for (int i = 0; i < NUM_PORTS; i++) begin
            automatic int expected_seg;
            expected_seg = (i / 4) % 8;
            if (k_coeff[i] !== k_expected[expected_seg] || b_intercept[i] !== b_expected[expected_seg]) begin
                $display("ERROR: Timing issue at port %0d", i);
                errors++;
            end
        end
        if (errors == 0) $display("  PASS: All outputs stable within 1ns (combinational)\n");

        // Summary
        $display("========================================================================");
        if (errors == 0) begin
            $display("                    ALL TESTS PASSED!");
        end else begin
            $display("                    TESTS FAILED: %0d errors", errors);
        end
        $display("========================================================================\n");

        $finish;
    end

endmodule
