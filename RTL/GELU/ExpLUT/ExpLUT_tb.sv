`timescale 1ns/1ps

module SharedLUT_tb;

    // Parameters
    localparam int Q = 26;          // Q5.26 format
    localparam int W = 32;          // 32-bit width
    localparam int NUM_SEGMENTS = 8;
    localparam int NUM_PORTS = 32;

    // DUT signals
    logic [2:0] segment_idx [NUM_PORTS-1:0];
    logic signed [W-1:0] k_coeff [NUM_PORTS-1:0];
    logic signed [W-1:0] b_intercept [NUM_PORTS-1:0];

    // Expected values (Q5.26 format)
    localparam logic signed [W-1:0] K_expected [0:7] = '{
        32'h02E57078, 32'h03288B9B, 32'h0371B996, 32'h03C18722,
        32'h04188DB7, 32'h047774AE, 32'h04DEF287, 32'h054FCE46
    };

    localparam logic signed [W-1:0] B_expected [0:7] = '{
        32'h04000000, 32'h03F79C9B, 32'h03E5511D, 32'h03C76408,
        32'h039BE0BD, 32'h03609063, 32'h0312F200, 32'h02B031B9
    };

    // Error tracking
    int errors;

    // DUT instantiation
    SharedLUT #(
        .Q(Q),
        .W(W),
        .NUM_SEGMENTS(NUM_SEGMENTS),
        .NUM_PORTS(NUM_PORTS)
    ) dut (
        .segment_index(segment_idx),
        .k_coeff(k_coeff),
        .b_intercept(b_intercept)
    );

    // ====================================================================
    // Test 1: Timing verification
    // ====================================================================
    task test_timing;
        int i;
        begin
            $display("\n=== Test 1: Timing Check ===");
            for (i = 0; i < NUM_PORTS; i++) begin
                segment_idx[i] = 3'd0;
            end
            
            #1; // Only 1ns!
            
            for (i = 0; i < NUM_PORTS; i++) begin
                if (k_coeff[i] !== K_expected[0] || b_intercept[i] !== B_expected[0]) begin
                    $display("ERROR: Port %0d not stable after 1ns", i);
                    $display("  Expected K: 0x%08h, Got: 0x%08h", K_expected[0], k_coeff[i]);
                    $display("  Expected B: 0x%08h, Got: 0x%08h", B_expected[0], b_intercept[i]);
                    errors++;
                end
            end
            
            if (errors == 0) 
                $display("✓ PASS: All outputs stable within 1ns");
        end
    endtask

    // ====================================================================
    // Test 2: 12-Layer BERT simulation
    // ====================================================================
    task test_layers;
        int layer, i, seg;
        begin
            $display("\n=== Test 2: 12-Layer BERT Simulation ===");
            for (layer = 0; layer < 12; layer++) begin
                for (i = 0; i < NUM_PORTS; i++) begin
                    segment_idx[i] = (layer + i) % NUM_SEGMENTS;
                end
                #10;
                
                // Verify first port
                seg = segment_idx[0];
                if (k_coeff[0] !== K_expected[seg] || b_intercept[0] !== B_expected[seg]) begin
                    $display("ERROR: Layer %0d port 0 mismatch", layer);
                    $display("  Segment: %0d", seg);
                    $display("  Expected K: 0x%08h, Got: 0x%08h", K_expected[seg], k_coeff[0]);
                    $display("  Expected B: 0x%08h, Got: 0x%08h", B_expected[seg], b_intercept[0]);
                    errors++;
                end
            end
            
            if (errors == 0)
                $display("✓ PASS: All 12 layers correct");
        end
    endtask

    // ====================================================================
    // Test 3: Random access pattern
    // ====================================================================
    task test_random;
        int i, seg;
        string status;
        begin
            $display("\n=== Test 3: Random Access Pattern ===");
            
            for (i = 0; i < NUM_PORTS; i++) begin
                segment_idx[i] = $urandom_range(0, 7);
            end
            #10;
            
            $display("Port | Seg |   k_coeff (hex)   |  b_intercept (hex) | Status");
            $display("-----|-----|-------------------|--------------------|---------");
            
            for (i = 0; i < 8; i++) begin  // Show first 8 ports
                seg = segment_idx[i];
                status = (k_coeff[i] === K_expected[seg] && 
                         b_intercept[i] === B_expected[seg]) ? "PASS" : "FAIL";
                $display("  %2d |  %0d  |     0x%08h    |     0x%08h     | %s", 
                         i, seg, k_coeff[i], b_intercept[i], status);
                
                if (status == "FAIL") errors++;
            end
            
            // Check remaining ports
            for (i = 8; i < NUM_PORTS; i++) begin
                seg = segment_idx[i];
                if (k_coeff[i] !== K_expected[seg] || b_intercept[i] !== B_expected[seg]) begin
                    errors++;
                end
            end
            
            if (errors == 0)
                $display("✓ PASS: Random access correct");
        end
    endtask

    // ====================================================================
    // Test 4: All segments via port 0 (with real number conversion)
    // ====================================================================
    task test_all_segments;
        int seg;
        real k_real, b_real;
        longint k_int, b_int;  // Use longint for 32-bit values
        begin
            $display("\n=== Test 4: All Segment Entries (Q5.26) ===");
            $display("Seg |      k (hex)        k (real)   |      b (hex)        b (real)");
            $display("----|----------------------------------|----------------------------------");
            
            for (seg = 0; seg < NUM_SEGMENTS; seg++) begin
                segment_idx[0] = seg;
                #10;
                
                // Convert Q5.26 to real: divide by 2^26
                k_int = k_coeff[0];
                b_int = b_intercept[0];
                k_real = $itor(k_int) / (2.0 ** Q);
                b_real = $itor(b_int) / (2.0 ** Q);
                
                $display(" %0d  |  0x%08h  %9.6f  |  0x%08h  %9.6f", 
                         seg, k_coeff[0], k_real, b_intercept[0], b_real);
                
                if (k_coeff[0] !== K_expected[seg] || b_intercept[0] !== B_expected[seg]) begin
                    $display("ERROR: Segment %0d mismatch!", seg);
                    $display("  Expected K: 0x%08h, Got: 0x%08h", K_expected[seg], k_coeff[0]);
                    $display("  Expected B: 0x%08h, Got: 0x%08h", B_expected[seg], b_intercept[0]);
                    errors++;
                end
            end
            
            if (errors == 0)
                $display("✓ PASS: All LUT entries correct");
        end
    endtask

    // ====================================================================
    // Test 5: Multi-port stress test (all ports, all segments)
    // ====================================================================
    task test_stress;
        int port, seg;
        int local_errors;
        begin
            $display("\n=== Test 5: Multi-Port Stress Test ===");
            $display("Testing all %0d ports with all %0d segments...", NUM_PORTS, NUM_SEGMENTS);
            
            local_errors = 0;
            
            for (seg = 0; seg < NUM_SEGMENTS; seg++) begin
                // Set all ports to same segment
                for (port = 0; port < NUM_PORTS; port++) begin
                    segment_idx[port] = seg;
                end
                #5;
                
                // Verify all ports
                for (port = 0; port < NUM_PORTS; port++) begin
                    if (k_coeff[port] !== K_expected[seg] || 
                        b_intercept[port] !== B_expected[seg]) begin
                        $display("ERROR: Segment %0d, Port %0d failed", seg, port);
                        local_errors++;
                        errors++;
                    end
                end
            end
            
            if (local_errors == 0)
                $display("✓ PASS: %0d segment × %0d port = %0d tests passed", 
                         NUM_SEGMENTS, NUM_PORTS, NUM_SEGMENTS * NUM_PORTS);
            else
                $display("✗ FAIL: %0d/%0d tests failed", 
                         local_errors, NUM_SEGMENTS * NUM_PORTS);
        end
    endtask

    // ====================================================================
    // Main test sequence
    // ====================================================================
    initial begin
        int j;
        
        errors = 0;
        
        $display("========================================");
        $display("  ExpLUT Testbench (Q5.26 Format)");
        $display("  WIDTH=%0d bits, Q=%0d fractional bits", W, Q);
        $display("  PORTS=%0d, SEGMENTS=%0d", NUM_PORTS, NUM_SEGMENTS);
        $display("  Range: [%0d, %0d]", -(1 << (W-Q-1)), (1 << (W-Q-1))-1);
        $display("  Precision: 2^-%0d ≈ %.3e", Q, 2.0**(-Q));
        $display("========================================");
        
        // Initialize
        for (j = 0; j < NUM_PORTS; j++) begin
            segment_idx[j] = 3'd0;
        end
        #10;
        
        // Run all tests
        test_timing();
        test_layers();
        test_random();
        test_all_segments();
        test_stress();  // Extra stress test for Q5.26
        
        // Summary
        $display("\n========================================");
        if (errors == 0)
            $display("  ✓ ALL TESTS PASSED!");
        else
            $display("  ✗ FAILED: %0d errors", errors);
        $display("========================================\n");
        
        $finish;
    end

endmodule
