`timescale 1ns / 1ps

module tb_quantize();

    // -----------------------------------------
    // Parameters
    // -----------------------------------------
    localparam DATAWIDTH_in = 32;
    localparam M_width      = 32; // Python M0 is up to 32 bits!
    localparam S_width      = 8;

    // -----------------------------------------
    // Shared Stimulus Signals
    // -----------------------------------------
    logic                                 clk;
    logic                                 rst_n;
    logic                                 valid_in;
    logic signed [DATAWIDTH_in-1:0]       data_in;
    logic        [M_width-1:0]            scale_M;
    logic signed [S_width-1:0]            scale_S; // Signed to allow negative shifts!

    // Outputs for INT8 DUT
    logic signed [7:0]                    data_out_int8;
    logic                                 valid_out_int8;

    // Outputs for Q5.26 (INT32) DUT
    logic signed [31:0]                   data_out_q5_26;
    logic                                 valid_out_q5_26;

    // -----------------------------------------
    // DUT 1: Standard INT8 Quantizer
    // -----------------------------------------
    quantize #(
        .DATAWIDTH_in(DATAWIDTH_in),
        .DATAWIDTH_out(8),             // 8-bit output
        .M_width(M_width),
        .S_width(S_width)
    ) dut_int8 (
        .clk(clk),
        .rst_n(rst_n),
        .valid_in(valid_in),
        .data_in(data_in),
        .scale_M(scale_M),
        .scale_S(scale_S),
        .data_out(data_out_int8),
        .valid_out(valid_out_int8)
    );

    // -----------------------------------------
    // DUT 2: Q5.26 (32-bit) Quantizer
    // -----------------------------------------
    quantize #(
        .DATAWIDTH_in(DATAWIDTH_in),
        .DATAWIDTH_out(32),            // 32-bit output!
        .M_width(M_width),
        .S_width(S_width)
    ) dut_q5_26 (
        .clk(clk),
        .rst_n(rst_n),
        .valid_in(valid_in),
        .data_in(data_in),
        .scale_M(scale_M),
        .scale_S(scale_S),
        .data_out(data_out_q5_26),
        .valid_out(valid_out_q5_26)
    );

    // -----------------------------------------
    // Clock Generation
    // -----------------------------------------
    initial begin
        clk = 0;
        forever #5 clk = ~clk; 
    end

    // -----------------------------------------
    // Main Test Sequence
    // -----------------------------------------
    initial begin
        rst_n    = 0;
        valid_in = 0;
        data_in  = 0;
        scale_M  = 0;
        scale_S  = 0;

        #20 rst_n = 1;
        $display("\n================ STARTING QUANTIZATION TESTS ================");

        // TC1: Normal INT8 scale down. 
        // M0 = 1073741824 is exactly 0.5 in fixed point (2^30 / 2^31)
        // 100 * 0.5 = 50. 
        @(negedge clk);
        apply_stimulus("TC1 - Normal Scale Down", 100, 1073741824, 0);

        // TC2: INT8 Clamping. 
        // 500 * 0.5 = 250. This exceeds 127, so INT8 should clamp, Q32 should pass.
        @(negedge clk);
        apply_stimulus("TC2 - INT8 Positive Clamp", 500, 1073741824, 0);

        // TC3: Normal Q5.26 Conversion (Negative Shift)
        // S = -15. Division becomes (31 - 15) = 16.
        // 1,000 * 1,073,741,824 = 1,073,741,824,000. 
        // 1,073,741,824,000 >> 16 = 16,384,000. 
        @(negedge clk);
        apply_stimulus("TC3 - Q5.26 Standard Conversion", 1000, 1073741824, -15);

        // TC4: Q5.26 Clamping (32-bit Overflow Prevention)
        // Data = 500,000. Same M and S as TC3.
        // Math yields 8,192,000,000. This exceeds 32-bit max (2,147,483,647).
        @(negedge clk);
        apply_stimulus("TC4 - Q5.26 32-bit Positive Clamp", 500000, 1073741824, -15);

        // TC5: Negative Value processing
        // -150 * 0.5 = -75.
        @(negedge clk);
        apply_stimulus("TC5 - Normal Negative Processing", -150, 1073741824, 0);

        // TC4.5: Q5.26 Normal Scale Up (NO CLAMPING)
        // Data = 5,000. M0 = 1073741824 (0.5). S = -15.
        // Math: Data * 0.5 * 2^15 = 5000 * 0.5 * 32768 = 81,920,000.
        // 81,920,000 is well below the 32-bit max of 2,147,483,647.
        @(negedge clk);
        apply_stimulus("TC4.5 - Q5.26 Normal Scale Up (No Clamp)", 5000, 1073741824, -15);

        #40;
        $display("================ TESTS COMPLETED ================\n");
        $finish;
    end


    // -----------------------------------------
    // Task: Apply Stimulus
    // -----------------------------------------
    task apply_stimulus(
        input string test_name,
        input logic signed [DATAWIDTH_in-1:0] in_val,
        input logic        [M_width-1:0]      m_val,
        input logic signed [S_width-1:0]      s_val
    );
        begin
            
            valid_in <= 1'b1;
            data_in  <= in_val;
            scale_M  <= m_val;
            scale_S  <= s_val;
            
            @(posedge clk);
            valid_in <= 1'b0; 

            // Wait for pipeline to finish (valid_out high)
            @(posedge valid_out_int8);
            $display("------------------------------------------------------------");
            $display("TEST CASE: %s", test_name);
            $display("Inputs -> Data: %0d | M0: %0d | S: %0d", in_val, m_val, s_val);
            $display("Output (INT8)   : %0d", data_out_int8);
            $display("Output (Q5.26)  : %0d", data_out_q5_26);
        end
    endtask

endmodule