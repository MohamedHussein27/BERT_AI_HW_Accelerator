`timescale 1ns/1ps

module LOD_tb;

    localparam int W = 64;  // Test with 64-bit for DU compatibility

    logic [W-1:0]          data_in;
    logic [$clog2(W)-1:0]  lod_pos;
    logic                  found;

    // Test statistics
    int passed = 0;
    int failed = 0;

    // Instantiate DUT
    LOD #(.W(W)) u_dut (
        .data_in (data_in),
        .lod_pos (lod_pos),
        .found   (found)
    );

    initial begin
        $display("\n=======================================================");
        $display(" Leading One Detector (LOD) Testbench");
        $display(" Width: %0d bits, LOD Position Width: %0d bits", W, $clog2(W));
        $display("=======================================================\n");

        // Test 1: All Zeros
        $display("[TEST 1] All Zeros");
        $display("Input                | Found | Position | Expected Pos | Status");
        $display("---------------------|-------|----------|--------------|--------");
        data_in = '0;
        #1;
        check_result("All zeros");

        // Test 2: LSB only
        $display("\n[TEST 2] LSB Only");
        data_in = {{(W-1){1'b0}}, 1'b1};
        #1;
        check_result("LSB=1");

        // Test 3: MSB only
        $display("\n[TEST 3] MSB Only");
        data_in = {1'b1, {(W-1){1'b0}}};
        #1;
        check_result("MSB=1");

        // Test 4: All ones
        $display("\n[TEST 4] All Ones");
        data_in = {(W){1'b1}};
        #1;
        check_result("All ones");

        // Test 5: Walking ones (power of 2)
        $display("\n[TEST 5] Walking Ones (Each bit position)");
        $display("Input                | Found | Position | Expected Pos | Status");
        $display("---------------------|-------|----------|--------------|--------");
        for (int i = 0; i < W; i++) begin
            data_in = (64'b1 << i);
            #1;
            check_result($sformatf("Bit[%0d]", i));
        end

        // Test 6: Specific patterns for Q48.16 values
        $display("\n[TEST 6] Q48.16 Representative Values");
        $display("Value (Q48.16)       | Binary (top 16) | Found | Position | Status");
        $display("---------------------|-----------------|-------|----------|--------");
        
        // 1.0 in Q48.16
        data_in = 64'h0000000000010000;
        #1;
        check_result_q48("1.0");
        
        // 2.0 in Q48.16
        data_in = 64'h0000000000020000;
        #1;
        check_result_q48("2.0");
        
        // 1024.0 in Q48.16
        data_in = 64'h0000000004000000;
        #1;
        check_result_q48("1024.0");
        
        // Large value (2^40)
        data_in = 64'h0000010000000000;
        #1;
        check_result_q48("2^40");
        
        // Max Q48.16 (near overflow)
        data_in = 64'h7FFFFFFFFFFFFFFF;
        #1;
        check_result_q48("Max");

        // Test 7: Patterns with multiple ones (MSB should be detected)
        $display("\n[TEST 7] Multiple Bits Set (MSB Detection)");
        $display("Input (hex)          | Found | Position | Expected Pos | Status");
        $display("---------------------|-------|----------|--------------|--------");
        data_in = 64'h0000000000000003;  // bits [1:0]
        #1;
        check_result("0x3");
        
        data_in = 64'h00000000000000FF;  // bits [7:0]
        #1;
        check_result("0xFF");
        
        data_in = 64'h000000000000FFFF;  // bits [15:0]
        #1;
        check_result("0xFFFF");
        
        data_in = 64'h8000000000000001;  // MSB and LSB
        #1;
        check_result("0x8000000000000001");

        // Test 8: Random patterns
        $display("\n[TEST 8] Random Patterns (100 iterations)");
        repeat (100) begin
            data_in = {$random, $random};
            #1;
            check_result_silent();
        end
        $display("Completed 100 random tests");

        // Test 9: Edge cases from EU output range
        $display("\n[TEST 9] Expected EU Output Values");
        $display("Description          | Input (hex)      | Found | Position | Status");
        $display("---------------------|------------------|-------|----------|--------");
        
        // Small exp value (exp(-10) ≈ 0.000045 in Q48.16)
        data_in = 64'h0000000000000003;
        #1;
        check_result_desc("exp(-10)");
        
        // Medium exp value (exp(0) = 1.0)
        data_in = 64'h0000000000010000;
        #1;
        check_result_desc("exp(0)");
        
        // Large exp value (exp(20))
        data_in = 64'h0000001000000000;
        #1;
        check_result_desc("exp(20)");
        
        // Very large exp value (exp(40))
        data_in = 64'h0100000000000000;
        #1;
        check_result_desc("exp(40)");

        // Summary
        $display("\n=======================================================");
        $display(" Test Summary");
        $display("=======================================================");
        $display(" Total Tests: %0d", passed + failed);
        $display(" Passed:      %0d", passed);
        $display(" Failed:      %0d", failed);
        
        if (failed == 0) begin
            $display("\n ✓ ALL TESTS PASSED!");
        end else begin
            $display("\n ✗ SOME TESTS FAILED!");
        end
        $display("=======================================================\n");
        
        $finish;
    end

    // Task: Check result with detailed output
    task check_result(string description);
        logic [$clog2(W)-1:0] exp_pos;
        logic                 exp_found;
        string status;
        
        // Golden Reference Model
        exp_found = |data_in;
        exp_pos   = '0;
        
        if (exp_found) begin
            for (int i = W-1; i >= 0; i--) begin
                if (data_in[i]) begin
                    exp_pos = i[$clog2(W)-1:0];
                    break;
                end
            end
        end

        // Compare
        if (found === exp_found && (!found || lod_pos === exp_pos)) begin
            status = "PASS";
            passed++;
        end else begin
            status = "FAIL";
            failed++;
        end

        $display("%-20s | %5b | %8d | %12d | %s", 
                 description, found, lod_pos, exp_pos, status);
        
        if (status == "FAIL") begin
            $display("  ERROR: Input=0x%h, Expected: found=%b pos=%0d, Got: found=%b pos=%0d",
                     data_in, exp_found, exp_pos, found, lod_pos);
        end
    endtask

    // Task: Check Q48.16 format values
    task check_result_q48(string value_str);
        logic [$clog2(W)-1:0] exp_pos;
        logic                 exp_found;
        string status;
        
        exp_found = |data_in;
        exp_pos   = '0;
        
        if (exp_found) begin
            for (int i = W-1; i >= 0; i--) begin
                if (data_in[i]) begin
                    exp_pos = i[$clog2(W)-1:0];
                    break;
                end
            end
        end

        if (found === exp_found && (!found || lod_pos === exp_pos)) begin
            status = "PASS";
            passed++;
        end else begin
            status = "FAIL";
            failed++;
        end

        $display("%-20s | %016b | %5b | %8d | %s", 
                 value_str, data_in[63:48], found, lod_pos, status);
    endtask

    // Task: Check with description
    task check_result_desc(string desc);
        logic [$clog2(W)-1:0] exp_pos;
        logic                 exp_found;
        string status;
        
        exp_found = |data_in;
        exp_pos   = '0;
        
        if (exp_found) begin
            for (int i = W-1; i >= 0; i--) begin
                if (data_in[i]) begin
                    exp_pos = i[$clog2(W)-1:0];
                    break;
                end
            end
        end

        if (found === exp_found && (!found || lod_pos === exp_pos)) begin
            status = "PASS";
            passed++;
        end else begin
            status = "FAIL";
            failed++;
        end

        $display("%-20s | 0x%016h | %5b | %8d | %s", 
                 desc, data_in, found, lod_pos, status);
    endtask

    // Task: Silent check for bulk testing
    task check_result_silent();
        logic [$clog2(W)-1:0] exp_pos;
        logic                 exp_found;
        
        exp_found = |data_in;
        exp_pos   = '0;
        
        if (exp_found) begin
            for (int i = W-1; i >= 0; i--) begin
                if (data_in[i]) begin
                    exp_pos = i[$clog2(W)-1:0];
                    break;
                end
            end
        end

        if (found === exp_found && (!found || lod_pos === exp_pos)) begin
            passed++;
        end else begin
            failed++;
            $display("  FAIL: Input=0x%h, Expected: found=%b pos=%0d, Got: found=%b pos=%0d",
                     data_in, exp_found, exp_pos, found, lod_pos);
        end
    endtask

endmodule
