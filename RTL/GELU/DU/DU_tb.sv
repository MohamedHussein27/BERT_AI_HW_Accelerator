`timescale 1ns/1ps

module DU_tb;

  parameter int Q = 26;
  parameter int W = 32;
  parameter int CLK_PERIOD = 10;

  logic                 clk;
  logic                 rst_n;
  logic                 valid_in;
  logic signed [W-1:0]  F;
  logic signed [W-1:0]  s_xi;
  logic                 valid_out;
  logic signed [W-1:0]  exponent;
  logic                 div_by_zero;
  logic                 result_sign;

  // Internal signals for monitoring
  logic signed [W-1:0] m1_stage3;
  logic signed [W-1:0] m2_stage3;
  logic signed [W-1:0] s1_stage3;
  logic signed [W-1:0] s2_stage3;

  DU #(.Q(Q), .W(W)) dut (
    .clk(clk),
    .rst_n(rst_n),
    .valid_in(valid_in),
    .F(F),
    .s_xi(s_xi),
    .valid_out(valid_out),
    .exponent(exponent),
    .div_by_zero(div_by_zero),
    .result_sign(result_sign)
  );

  // Probe internal signals
  assign m1_stage3 = dut.m1_stage3;
  assign m2_stage3 = dut.m2_stage3;
  assign s1_stage3 = dut.s1_stage3;
  assign s2_stage3 = dut.s2_stage3;

  // Helper function: convert Q5.26 to decimal
  function real q5_26_to_decimal(logic signed [W-1:0] val);
    begin
      q5_26_to_decimal = real'(val) / (real'(2) ** Q);
    end
  endfunction

  // Helper function: 2^exponent where exponent is in Q5.26
  function real compute_2_to_exponent(logic signed [W-1:0] exp_q5_26);
    real exp_decimal;
    begin
      exp_decimal = q5_26_to_decimal(exp_q5_26);
      compute_2_to_exponent = 2.0 ** exp_decimal;
    end
  endfunction

  // Helper function: calculate error percentage
  function real calc_error_percent(real expected, real actual);
    begin
      if (expected == 0.0)
        calc_error_percent = 0.0;
      else
        calc_error_percent = (actual - expected) / expected * 100.0;
    end
  endfunction

  // Helper function: convert to binary
  function string to_binary_q5_26(logic signed [W-1:0] val);
    begin
      return $sformatf("%b", val);
    end
  endfunction

  // Clock generation
  initial begin
    clk = 0;
    forever #(CLK_PERIOD/2) clk = ~clk;
  end

  // Test stimulus
  initial begin
    rst_n = 0;
    valid_in = 0;
    F = 0;
    s_xi = 0;

    #(CLK_PERIOD*2);
    rst_n = 1;
    #(CLK_PERIOD*2);

    // ========== TEST 1: 7.0 / 3.0 = 2.333... ==========
    $display("\n╔════════════════════════════════════════════════════════════════════╗");
    $display("║ TEST 1: F = 7.0 / s_xi = 3.0 (Expected = 2.333333...)           ║");
    $display("╚════════════════════════════════════════════════════════════════════╝");
    
    valid_in = 1;
    F = 32'h1C000000;     // 7.0 in Q5.26
    s_xi = 32'h0C000000;  // 3.0 in Q5.26
    
    $display("\nGOLDEN MODEL (Decimal):");
    $display("  F_decimal     = %.10f", q5_26_to_decimal(F));
    $display("  s_xi_decimal  = %.10f", q5_26_to_decimal(s_xi));
    $display("  Expected Result (7.0/3.0) = %.10f", q5_26_to_decimal(F) / q5_26_to_decimal(s_xi));
    
    #CLK_PERIOD;
    valid_in = 0;
    
    wait(valid_out);
    #CLK_PERIOD;
    
    $display("\nFINAL OUTPUT:");
    $display("  exponent (decimal) = %.10f", q5_26_to_decimal(exponent));
    $display("  2^exponent = %.10f", compute_2_to_exponent(exponent));
    $display("  Expected:  %.10f", q5_26_to_decimal(F) / q5_26_to_decimal(s_xi));
    $display("  Error Percentage: %.6f%%", calc_error_percent(q5_26_to_decimal(F) / q5_26_to_decimal(s_xi), compute_2_to_exponent(exponent)));
    #(CLK_PERIOD*2);

    // ========== TEST 2: 11.0 / 7.0 = 1.571428... ==========
    $display("\n╔════════════════════════════════════════════════════════════════════╗");
    $display("║ TEST 2: F = 11.0 / s_xi = 7.0 (Expected = 1.571428...)          ║");
    $display("╚════════════════════════════════════════════════════════════════════╝");
    
    valid_in = 1;
    F = 32'h2C000000;     // 11.0 in Q5.26
    s_xi = 32'h1C000000;  // 7.0 in Q5.26
    
    $display("\nGOLDEN MODEL (Decimal):");
    $display("  F_decimal     = %.10f", q5_26_to_decimal(F));
    $display("  s_xi_decimal  = %.10f", q5_26_to_decimal(s_xi));
    $display("  Expected Result (11.0/7.0) = %.10f", q5_26_to_decimal(F) / q5_26_to_decimal(s_xi));
    
    #CLK_PERIOD;
    valid_in = 0;
    
    wait(valid_out);
    #CLK_PERIOD;
    
    $display("\nFINAL OUTPUT:");
    $display("  exponent (decimal) = %.10f", q5_26_to_decimal(exponent));
    $display("  2^exponent = %.10f", compute_2_to_exponent(exponent));
    $display("  Expected:  %.10f", q5_26_to_decimal(F) / q5_26_to_decimal(s_xi));
    $display("  Error Percentage: %.6f%%", calc_error_percent(q5_26_to_decimal(F) / q5_26_to_decimal(s_xi), compute_2_to_exponent(exponent)));
    #(CLK_PERIOD*2);

    // ========== TEST 3: 13.0 / 5.0 = 2.6 ==========
    $display("\n╔════════════════════════════════════════════════════════════════════╗");
    $display("║ TEST 3: F = 13.0 / s_xi = 5.0 (Expected = 2.6)                  ║");
    $display("╚════════════════════════════════════════════════════════════════════╝");
    
    valid_in = 1;
    F = 32'h34000000;     // 13.0 in Q5.26
    s_xi = 32'h14000000;  // 5.0 in Q5.26
    
    $display("\nGOLDEN MODEL (Decimal):");
    $display("  F_decimal     = %.10f", q5_26_to_decimal(F));
    $display("  s_xi_decimal  = %.10f", q5_26_to_decimal(s_xi));
    $display("  Expected Result (13.0/5.0) = %.10f", q5_26_to_decimal(F) / q5_26_to_decimal(s_xi));
    
    #CLK_PERIOD;
    valid_in = 0;
    
    wait(valid_out);
    #CLK_PERIOD;
    
    $display("\nFINAL OUTPUT:");
    $display("  exponent (decimal) = %.10f", q5_26_to_decimal(exponent));
    $display("  2^exponent = %.10f", compute_2_to_exponent(exponent));
    $display("  Expected:  %.10f", q5_26_to_decimal(F) / q5_26_to_decimal(s_xi));
    $display("  Error Percentage: %.6f%%", calc_error_percent(q5_26_to_decimal(F) / q5_26_to_decimal(s_xi), compute_2_to_exponent(exponent)));
    #(CLK_PERIOD*2);

    // ========== TEST 4: 19.0 / 6.0 = 3.166666... ==========
    $display("\n╔════════════════════════════════════════════════════════════════════╗");
    $display("║ TEST 4: F = 19.0 / s_xi = 6.0 (Expected = 3.166666...)          ║");
    $display("╚════════════════════════════════════════════════════════════════════╝");
    
    valid_in = 1;
    F = 32'h4C000000;     // 19.0 in Q5.26
    s_xi = 32'h18000000;  // 6.0 in Q5.26
    
    $display("\nGOLDEN MODEL (Decimal):");
    $display("  F_decimal     = %.10f", q5_26_to_decimal(F));
    $display("  s_xi_decimal  = %.10f", q5_26_to_decimal(s_xi));
    $display("  Expected Result (19.0/6.0) = %.10f", q5_26_to_decimal(F) / q5_26_to_decimal(s_xi));
    
    #CLK_PERIOD;
    valid_in = 0;
    
    wait(valid_out);
    #CLK_PERIOD;
    
    $display("\nFINAL OUTPUT:");
    $display("  exponent (decimal) = %.10f", q5_26_to_decimal(exponent));
    $display("  2^exponent = %.10f", compute_2_to_exponent(exponent));
    $display("  Expected:  %.10f", q5_26_to_decimal(F) / q5_26_to_decimal(s_xi));
    $display("  Error Percentage: %.6f%%", calc_error_percent(q5_26_to_decimal(F) / q5_26_to_decimal(s_xi), compute_2_to_exponent(exponent)));
    #(CLK_PERIOD*2);

    // ========== TEST 5: 23.0 / 9.0 = 2.555555... ==========
    $display("\n╔════════════════════════════════════════════════════════════════════╗");
    $display("║ TEST 5: F = 23.0 / s_xi = 9.0 (Expected = 2.555555...)          ║");
    $display("╚════════════════════════════════════════════════════════════════════╝");
    
    valid_in = 1;
    F = 32'h5C000000;     // 23.0 in Q5.26
    s_xi = 32'h24000000;  // 9.0 in Q5.26
    
    $display("\nGOLDEN MODEL (Decimal):");
    $display("  F_decimal     = %.10f", q5_26_to_decimal(F));
    $display("  s_xi_decimal  = %.10f", q5_26_to_decimal(s_xi));
    $display("  Expected Result (23.0/9.0) = %.10f", q5_26_to_decimal(F) / q5_26_to_decimal(s_xi));
    
    #CLK_PERIOD;
    valid_in = 0;
    
    wait(valid_out);
    #CLK_PERIOD;
    
    $display("\nFINAL OUTPUT:");
    $display("  exponent (decimal) = %.10f", q5_26_to_decimal(exponent));
    $display("  2^exponent = %.10f", compute_2_to_exponent(exponent));
    $display("  Expected:  %.10f", q5_26_to_decimal(F) / q5_26_to_decimal(s_xi));
    $display("  Error Percentage: %.6f%%", calc_error_percent(q5_26_to_decimal(F) / q5_26_to_decimal(s_xi), compute_2_to_exponent(exponent)));
    #(CLK_PERIOD*2);

    // ========== TEST 6: 31.0 / 8.0 = 3.875 ==========
    $display("\n╔════════════════════════════════════════════════════════════════════╗");
    $display("║ TEST 6: F = 31.0 / s_xi = 8.0 (Expected = 3.875)                ║");
    $display("╚════════════════════════════════════════════════════════════════════╝");
    
    valid_in = 1;
    F = 32'h7C000000;     // 31.0 in Q5.26
    s_xi = 32'h20000000;  // 8.0 in Q5.26
    
    $display("\nGOLDEN MODEL (Decimal):");
    $display("  F_decimal     = %.10f", q5_26_to_decimal(F));
    $display("  s_xi_decimal  = %.10f", q5_26_to_decimal(s_xi));
    $display("  Expected Result (31.0/8.0) = %.10f", q5_26_to_decimal(F) / q5_26_to_decimal(s_xi));
    
    #CLK_PERIOD;
    valid_in = 0;
    
    wait(valid_out);
    #CLK_PERIOD;
    
    $display("\nFINAL OUTPUT:");
    $display("  exponent (decimal) = %.10f", q5_26_to_decimal(exponent));
    $display("  2^exponent = %.10f", compute_2_to_exponent(exponent));
    $display("  Expected:  %.10f", q5_26_to_decimal(F) / q5_26_to_decimal(s_xi));
    $display("  Error Percentage: %.6f%%", calc_error_percent(q5_26_to_decimal(F) / q5_26_to_decimal(s_xi), compute_2_to_exponent(exponent)));
    #(CLK_PERIOD*2);

    // ========== TEST 7: -17.0 / 4.0 = -4.25 (negative with fraction) ==========
    $display("\n╔════════════════════════════════════════════════════════════════════╗");
    $display("║ TEST 7: F = -17.0 / s_xi = 4.0 (Expected = -4.25)               ║");
    $display("╚════════════════════════════════════════════════════════════════════╝");
    
    valid_in = 1;
    F = -32'h44000000;    // -17.0 in Q5.26
    s_xi = 32'h10000000;  // 4.0 in Q5.26
    
    $display("\nGOLDEN MODEL (Decimal):");
    $display("  F_decimal     = %.10f", q5_26_to_decimal(F));
    $display("  s_xi_decimal  = %.10f", q5_26_to_decimal(s_xi));
    $display("  Expected Result (-17.0/4.0) = %.10f", q5_26_to_decimal(F) / q5_26_to_decimal(s_xi));
    $display("  Expected Magnitude: %.10f", q5_26_to_decimal(-F) / q5_26_to_decimal(s_xi));
    
    #CLK_PERIOD;
    valid_in = 0;
    
    wait(valid_out);
    #CLK_PERIOD;
    
    $display("\nFINAL OUTPUT:");
    $display("  exponent (decimal) = %.10f", q5_26_to_decimal(exponent));
    $display("  2^exponent = %.10f", compute_2_to_exponent(exponent));
    $display("  Expected magnitude: %.10f", q5_26_to_decimal(-F) / q5_26_to_decimal(s_xi));
    $display("  result_sign = %b (1=negative)", result_sign);
    $display("  Error Percentage: %.6f%%", calc_error_percent(q5_26_to_decimal(-F) / q5_26_to_decimal(s_xi), compute_2_to_exponent(exponent)));
    #(CLK_PERIOD*2);

    $display("\n╔════════════════════════════════════════════════════════════════════╗");
    $display("║ SIMULATION COMPLETE                                               ║");
    $display("╚════════════════════════════════════════════════════════════════════╝\n");
    $finish;
  end

endmodule
