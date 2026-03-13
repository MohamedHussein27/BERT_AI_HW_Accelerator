`timescale 1ns / 1ps

module tb_layer0_quantization();

    // -----------------------------------------
    // Global Parameters
    // -----------------------------------------
    localparam M_WIDTH = 32;
    localparam S_WIDTH = 8;

    logic clk;
    logic rst_n;

    // -----------------------------------------
    // DUT 1: Standard 32->8 Quantizer
    // -----------------------------------------
    logic valid_in_int8, valid_out_int8;
    logic signed [31:0] data_in_int8;
    logic [M_WIDTH-1:0] scale_m_int8;
    logic signed [S_WIDTH-1:0] scale_s_int8;
    logic signed [7:0] data_out_int8;

    quantize #(
        .DATAWIDTH_in(32), .DATAWIDTH_out(8), .M_width(M_WIDTH), .S_width(S_WIDTH)
    ) dut_int8 (
        .clk(clk), .rst_n(rst_n), .valid_in(valid_in_int8), .data_in(data_in_int8),
        .scale_M(scale_m_int8), .scale_S(scale_s_int8), .data_out(data_out_int8), .valid_out(valid_out_int8)
    );

    // -----------------------------------------
    // DUT 2: 32->32 Dequantizer (Q5.26 / Q10.22)
    // -----------------------------------------
    logic valid_in_q32, valid_out_q32;
    logic signed [31:0] data_in_q32;
    logic [M_WIDTH-1:0] scale_m_q32;
    logic signed [S_WIDTH-1:0] scale_s_q32;
    logic signed [31:0] data_out_q32;

    quantize #(
        .DATAWIDTH_in(32), .DATAWIDTH_out(32), .M_width(M_WIDTH), .S_width(S_WIDTH)
    ) dut_q32 (
        .clk(clk), .rst_n(rst_n), .valid_in(valid_in_q32), .data_in(data_in_q32),
        .scale_M(scale_m_q32), .scale_S(scale_s_q32), .data_out(data_out_q32), .valid_out(valid_out_q32)
    );

    // -----------------------------------------
    // DUT 3: 64->8 Quantizer (GELU Q48.16)
    // -----------------------------------------
    logic valid_in_gelu, valid_out_gelu;
    logic signed [63:0] data_in_gelu;
    logic [M_WIDTH-1:0] scale_m_gelu;
    logic signed [S_WIDTH-1:0] scale_s_gelu;
    logic signed [7:0] data_out_gelu;

    quantize #(
        .DATAWIDTH_in(64), .DATAWIDTH_out(8), .M_width(M_WIDTH), .S_width(S_WIDTH)
    ) dut_gelu (
        .clk(clk), .rst_n(rst_n), .valid_in(valid_in_gelu), .data_in(data_in_gelu),
        .scale_M(scale_m_gelu), .scale_S(scale_s_gelu), .data_out(data_out_gelu), .valid_out(valid_out_gelu)
    );

    // -----------------------------------------
    // Clock Generation
    // -----------------------------------------
    initial begin
        clk = 0;
        forever #5 clk = ~clk; 
    end

    // -----------------------------------------
    // Tasks to drive stimulus to specific DUTs
    // -----------------------------------------
    task run_int8(input string name, input logic signed [31:0] din, input logic [31:0] m0, input logic signed [7:0] s);
        @(negedge clk);
        valid_in_int8 <= 1'b1; data_in_int8 <= din; scale_m_int8 <= m0; scale_s_int8 <= s;
        @(negedge clk);
        valid_in_int8 <= 1'b0;
        @(posedge valid_out_int8);
        $display("[%-25s] IN: %10d | M0: %10d | S: %3d | OUT (8-bit): %4d", name, din, m0, s, data_out_int8);
    endtask

    task run_q32(input string name, input logic signed [31:0] din, input logic [31:0] m0, input logic signed [7:0] s);
        @(negedge clk);
        valid_in_q32 <= 1'b1; data_in_q32 <= din; scale_m_q32 <= m0; scale_s_q32 <= s;
        @(negedge clk);
        valid_in_q32 <= 1'b0;
        @(posedge valid_out_q32);
        $display("[%-25s] IN: %10d | M0: %10d | S: %3d | OUT (32-bit): %10d", name, din, m0, s, data_out_q32);
    endtask

    task run_gelu(input string name, input logic signed [63:0] din, input logic [31:0] m0, input logic signed [7:0] s);
        @(negedge clk);
        valid_in_gelu <= 1'b1; data_in_gelu <= din; scale_m_gelu <= m0; scale_s_gelu <= s;
        @(negedge clk);
        valid_in_gelu <= 1'b0;
        @(posedge valid_out_gelu);
        $display("[%-25s] IN: %10d | M0: %10d | S: %3d | OUT (8-bit): %4d", name, din, m0, s, data_out_gelu);
    endtask

    // -----------------------------------------
    // Main Sequence: LAYER 0 PIPELINE
    // -----------------------------------------
    initial begin
        // Init
        rst_n = 0;
        valid_in_int8 = 0; valid_in_q32 = 0; valid_in_gelu = 0;
        #20 rst_n = 1; #10;

        $display("\n================ LAYER 0 QUANTIZATION DATAPATH VERIFICATION ================\n");

        // 1. QKV Projections (Assuming raw accumulator value of ~150,000)
        run_int8("ATTEN-Q Quant (32->8)", 150000, 1972133394, 11);
        run_int8("ATTEN-K Quant (32->8)", 150000, 1104008503, 10);
        run_int8("ATTEN-V Quant (32->8)", 150000, 1958868565, 10);

        // 2. QKt to Softmax (Scaling an attention score of 2,500 up into Q5.26)
        run_q32("QKt Dequant (32->Q5.26)", 2500, 2051840830, -12);

        // 3. Softmax Output (Inputting 16384, which is 0.5 in Q1.15 format)
        run_int8("Softmax Quant (Q1.15->8)", 16384, 1417471555, 5);

        // 4. Context (.VGeMM) and Output (Wo)
        run_int8(".VGeMM Quant (32->8)", 85000, 2130779236, 8);
        run_int8("Wo Quantize (32->8)", 120000, 1614228903, 10);

        // 5. Add & Norm (Assuming the Q-format was resolved, passing a simulated normalized value)
        run_int8("Add&Norm Quant (?->8)", 500000, 1845025262, 21);

        // 6. FFN1 to GELU (Scaling an FFN MAC of 8,500 into Q10.22)
        run_q32("FFN1 Dequant (32->Q10.22)", 8500, 1449841874, -6);

        // 7. GELU Output to FFN2 
        // Inputting 327680 into the 64-bit DUT, which is exactly 5.0 in Q48.16 format (5 * 2^16)
        run_gelu("GELU Quant (Q48.16->8)", 64'd327680, 1556097776, 10);

        // 8. FFN2 Final Output
        run_int8("FFN2 Quant (32->8)", 95000, 1763007950, 11);

        #50;
        $display("\n================ DATAPATH VERIFICATION COMPLETE ================\n");
        $finish;
    end
endmodule