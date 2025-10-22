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
  logic                 result_sign;

  // Internal signals for monitoring
  logic signed [W-1:0] denominator_stage0;
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
    .result_sign(result_sign)
  );

  // Probe internal signals
  assign denominator_stage0 = dut.denominator_stage0;
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

    $display("\n╔═══════════════════════════════════════════════════════════════════════╗");
    $display("║           DU Testbench - Testing F / (1 + s_xi)                    ║");
    $display("║           Where DU internally adds 1.0 (0x04000000) to s_xi        ║");
    $display("╚═══════════════════════════════════════════════════════════════════════╝\n");

    // ========== TEST 1: 7.0 / (1 + 2.0) = 7.0 / 3.0 = 2.333... ==========
    $display("\n╔════════════════════════════════════════════════════════════════════╗");
    $display("║ TEST 1: F = 7.0, s_xi = 2.0 → F/(1+s_xi) = 7.0/3.0 = 2.333...   ║");
    $display("╚════════════════════════════════════════════════════════════════════╝");
    
    valid_in = 1;
    F = 32'h00000000;     // 7.0 in Q5.26
    s_xi = 32'h04000000;  // 2.0 in Q5.26
    
    $display("\nINPUTS:");
    $display("  F        = 0x%08h = %.10f", F, q5_26_to_decimal(F));
    $display("  s_xi     = 0x%08h = %.10f", s_xi, q5_26_to_decimal(s_xi));
    $display("  Expected denominator (1 + s_xi) = %.10f", 1.0 + q5_26_to_decimal(s_xi));
    $display("  Expected result: %.10f / %.10f = %.10f", 
             q5_26_to_decimal(F), 1.0 + q5_26_to_decimal(s_xi),
             q5_26_to_decimal(F) / (1.0 + q5_26_to_decimal(s_xi)));
    
    #CLK_PERIOD;
    valid_in = 0;
    
    // Monitor denominator after stage 0
    #CLK_PERIOD;
    $display("\nINTERNAL (Stage 0):");
    $display("  denominator_stage0 = 0x%08h = %.10f", 
             denominator_stage0, q5_26_to_decimal(denominator_stage0));
    
    wait(valid_out);
    #CLK_PERIOD;
    
    $display("\nFINAL OUTPUT:");
    $display("  exponent (hex)     = 0x%08h", exponent);
    $display("  exponent (decimal) = %.10f", q5_26_to_decimal(exponent));
    $display("  2^exponent         = %.10f", compute_2_to_exponent(exponent));
    $display("  Expected           = %.10f", q5_26_to_decimal(F) / (1.0 + q5_26_to_decimal(s_xi)));
    $display("  Error              = %.6f%%", 
             calc_error_percent(q5_26_to_decimal(F) / (1.0 + q5_26_to_decimal(s_xi)), 
                               compute_2_to_exponent(exponent)));
    #(CLK_PERIOD*2);

    // ========== TEST 2: 11.0 / (1 + 6.0) = 11.0 / 7.0 = 1.571428... ==========
    $display("\n╔════════════════════════════════════════════════════════════════════╗");
    $display("║ TEST 2: F = 11.0, s_xi = 6.0 → F/(1+s_xi) = 11.0/7.0 = 1.571... ║");
    $display("╚════════════════════════════════════════════════════════════════════╝");
    
    valid_in = 1;
    F = 32'h2C000000;     // 11.0 in Q5.26
    s_xi = 32'h18000000;  // 6.0 in Q5.26
    
    $display("\nINPUTS:");
    $display("  F        = 0x%08h = %.10f", F, q5_26_to_decimal(F));
    $display("  s_xi     = 0x%08h = %.10f", s_xi, q5_26_to_decimal(s_xi));
    $display("  Expected denominator (1 + s_xi) = %.10f", 1.0 + q5_26_to_decimal(s_xi));
    $display("  Expected result: %.10f / %.10f = %.10f", 
             q5_26_to_decimal(F), 1.0 + q5_26_to_decimal(s_xi),
             q5_26_to_decimal(F) / (1.0 + q5_26_to_decimal(s_xi)));
    
    #CLK_PERIOD;
    valid_in = 0;
    
    #CLK_PERIOD;
    $display("\nINTERNAL (Stage 0):");
    $display("  denominator_stage0 = 0x%08h = %.10f", 
             denominator_stage0, q5_26_to_decimal(denominator_stage0));
    
    wait(valid_out);
    #CLK_PERIOD;
    
    $display("\nFINAL OUTPUT:");
    $display("  exponent (hex)     = 0x%08h", exponent);
    $display("  exponent (decimal) = %.10f", q5_26_to_decimal(exponent));
    $display("  2^exponent         = %.10f", compute_2_to_exponent(exponent));
    $display("  Expected           = %.10f", q5_26_to_decimal(F) / (1.0 + q5_26_to_decimal(s_xi)));
    $display("  Error              = %.6f%%", 
             calc_error_percent(q5_26_to_decimal(F) / (1.0 + q5_26_to_decimal(s_xi)), 
                               compute_2_to_exponent(exponent)));
    #(CLK_PERIOD*2);

    // ========== TEST 3: 13.0 / (1 + 4.0) = 13.0 / 5.0 = 2.6 ==========
    $display("\n╔════════════════════════════════════════════════════════════════════╗");
    $display("║ TEST 3: F = 13.0, s_xi = 4.0 → F/(1+s_xi) = 13.0/5.0 = 2.6      ║");
    $display("╚════════════════════════════════════════════════════════════════════╝");
    
    valid_in = 1;
    F = 32'h34000000;     // 13.0 in Q5.26
    s_xi = 32'h10000000;  // 4.0 in Q5.26
    
    $display("\nINPUTS:");
    $display("  F        = 0x%08h = %.10f", F, q5_26_to_decimal(F));
    $display("  s_xi     = 0x%08h = %.10f", s_xi, q5_26_to_decimal(s_xi));
    $display("  Expected denominator (1 + s_xi) = %.10f", 1.0 + q5_26_to_decimal(s_xi));
    $display("  Expected result: %.10f / %.10f = %.10f", 
             q5_26_to_decimal(F), 1.0 + q5_26_to_decimal(s_xi),
             q5_26_to_decimal(F) / (1.0 + q5_26_to_decimal(s_xi)));
    
    #CLK_PERIOD;
    valid_in = 0;
    
    #CLK_PERIOD;
    $display("\nINTERNAL (Stage 0):");
    $display("  denominator_stage0 = 0x%08h = %.10f", 
             denominator_stage0, q5_26_to_decimal(denominator_stage0));
    
    wait(valid_out);
    #CLK_PERIOD;
    
    $display("\nFINAL OUTPUT:");
    $display("  exponent (hex)     = 0x%08h", exponent);
    $display("  exponent (decimal) = %.10f", q5_26_to_decimal(exponent));
    $display("  2^exponent         = %.10f", compute_2_to_exponent(exponent));
    $display("  Expected           = %.10f", q5_26_to_decimal(F) / (1.0 + q5_26_to_decimal(s_xi)));
    $display("  Error              = %.6f%%", 
             calc_error_percent(q5_26_to_decimal(F) / (1.0 + q5_26_to_decimal(s_xi)), 
                               compute_2_to_exponent(exponent)));
    #(CLK_PERIOD*2);

    // ========== TEST 4: Small values - 0.5 / (1 + 0.2) = 0.5 / 1.2 = 0.4166... ==========
    $display("\n╔════════════════════════════════════════════════════════════════════╗");
    $display("║ TEST 4: F = 0.5, s_xi = 0.2 → F/(1+s_xi) = 0.5/1.2 = 0.4166...  ║");
    $display("╚════════════════════════════════════════════════════════════════════╝");
    
    valid_in = 1;
    F = 32'h02000000;     // 0.5 in Q5.26
    s_xi = 32'h00CCCCD0;  // ~0.2 in Q5.26
    
    $display("\nINPUTS:");
    $display("  F        = 0x%08h = %.10f", F, q5_26_to_decimal(F));
    $display("  s_xi     = 0x%08h = %.10f", s_xi, q5_26_to_decimal(s_xi));
    $display("  Expected denominator (1 + s_xi) = %.10f", 1.0 + q5_26_to_decimal(s_xi));
    $display("  Expected result: %.10f / %.10f = %.10f", 
             q5_26_to_decimal(F), 1.0 + q5_26_to_decimal(s_xi),
             q5_26_to_decimal(F) / (1.0 + q5_26_to_decimal(s_xi)));
    
    #CLK_PERIOD;
    valid_in = 0;
    
    #CLK_PERIOD;
    $display("\nINTERNAL (Stage 0):");
    $display("  denominator_stage0 = 0x%08h = %.10f", 
             denominator_stage0, q5_26_to_decimal(denominator_stage0));
    
    wait(valid_out);
    #CLK_PERIOD;
    
    $display("\nFINAL OUTPUT:");
    $display("  exponent (hex)     = 0x%08h", exponent);
    $display("  exponent (decimal) = %.10f", q5_26_to_decimal(exponent));
    $display("  2^exponent         = %.10f", compute_2_to_exponent(exponent));
    $display("  Expected           = %.10f", q5_26_to_decimal(F) / (1.0 + q5_26_to_decimal(s_xi)));
    $display("  Error              = %.6f%%", 
             calc_error_percent(q5_26_to_decimal(F) / (1.0 + q5_26_to_decimal(s_xi)), 
                               compute_2_to_exponent(exponent)));
    #(CLK_PERIOD*2);

    // ========== TEST 5: Negative numerator - -17.0 / (1 + 3.0) = -17.0 / 4.0 = -4.25 ==========
    $display("\n╔════════════════════════════════════════════════════════════════════╗");
    $display("║ TEST 5: F = -17.0, s_xi = 3.0 → F/(1+s_xi) = -17.0/4.0 = -4.25  ║");
    $display("╚════════════════════════════════════════════════════════════════════╝");
    
    valid_in = 1;
    F = -32'h44000000;    // -17.0 in Q5.26
    s_xi = 32'h0C000000;  // 3.0 in Q5.26
    
    $display("\nINPUTS:");
    $display("  F        = 0x%08h = %.10f", F, q5_26_to_decimal(F));
    $display("  s_xi     = 0x%08h = %.10f", s_xi, q5_26_to_decimal(s_xi));
    $display("  Expected denominator (1 + s_xi) = %.10f", 1.0 + q5_26_to_decimal(s_xi));
    $display("  Expected result: %.10f / %.10f = %.10f", 
             q5_26_to_decimal(F), 1.0 + q5_26_to_decimal(s_xi),
             q5_26_to_decimal(F) / (1.0 + q5_26_to_decimal(s_xi)));
    $display("  Expected magnitude: %.10f", 
             q5_26_to_decimal(-F) / (1.0 + q5_26_to_decimal(s_xi)));
    
    #CLK_PERIOD;
    valid_in = 0;
    
    #CLK_PERIOD;
    $display("\nINTERNAL (Stage 0):");
    $display("  denominator_stage0 = 0x%08h = %.10f", 
             denominator_stage0, q5_26_to_decimal(denominator_stage0));
    
    wait(valid_out);
    #CLK_PERIOD;
    
    $display("\nFINAL OUTPUT:");
    $display("  exponent (hex)     = 0x%08h", exponent);
    $display("  exponent (decimal) = %.10f", q5_26_to_decimal(exponent));
    $display("  2^exponent         = %.10f", compute_2_to_exponent(exponent));
    $display("  result_sign        = %b (1=negative)", result_sign);
    $display("  Expected magnitude = %.10f", q5_26_to_decimal(-F) / (1.0 + q5_26_to_decimal(s_xi)));
    $display("  Error              = %.6f%%", 
             calc_error_percent(q5_26_to_decimal(-F) / (1.0 + q5_26_to_decimal(s_xi)), 
                               compute_2_to_exponent(exponent)));
    #(CLK_PERIOD*2);

    // ========== TEST 6: Edge case - s_xi = 0, so 1 + 0 = 1 ==========
    $display("\n╔════════════════════════════════════════════════════════════════════╗");
    $display("║ TEST 6: F = 8.0, s_xi = 0.0 → F/(1+s_xi) = 8.0/1.0 = 8.0        ║");
    $display("╚════════════════════════════════════════════════════════════════════╝");
    
    valid_in = 1;
    F = 32'h20000000;     // 8.0 in Q5.26
    s_xi = 32'h00000000;  // 0.0 in Q5.26
    
    $display("\nINPUTS:");
    $display("  F        = 0x%08h = %.10f", F, q5_26_to_decimal(F));
    $display("  s_xi     = 0x%08h = %.10f", s_xi, q5_26_to_decimal(s_xi));
    $display("  Expected denominator (1 + s_xi) = %.10f", 1.0 + q5_26_to_decimal(s_xi));
    $display("  Expected result: %.10f / %.10f = %.10f", 
             q5_26_to_decimal(F), 1.0 + q5_26_to_decimal(s_xi),
             q5_26_to_decimal(F) / (1.0 + q5_26_to_decimal(s_xi)));
    
    #CLK_PERIOD;
    valid_in = 0;
    
    #CLK_PERIOD;
    $display("\nINTERNAL (Stage 0):");
    $display("  denominator_stage0 = 0x%08h = %.10f (should be 1.0 = 0x04000000)", 
             denominator_stage0, q5_26_to_decimal(denominator_stage0));
    
    wait(valid_out);
    #CLK_PERIOD;
    
    $display("\nFINAL OUTPUT:");
    $display("  exponent (hex)     = 0x%08h", exponent);
    $display("  exponent (decimal) = %.10f", q5_26_to_decimal(exponent));
    $display("  2^exponent         = %.10f", compute_2_to_exponent(exponent));
    $display("  Expected           = %.10f", q5_26_to_decimal(F) / (1.0 + q5_26_to_decimal(s_xi)));
    $display("  Error              = %.6f%%", 
             calc_error_percent(q5_26_to_decimal(F) / (1.0 + q5_26_to_decimal(s_xi)), 
                               compute_2_to_exponent(exponent)));
    #(CLK_PERIOD*2);

    $display("\n╔════════════════════════════════════════════════════════════════════╗");
    $display("║ SIMULATION COMPLETE - All tests verify F / (1 + s_xi)           ║");
    $display("╚════════════════════════════════════════════════════════════════════╝\n");
    $finish;
  end

endmodule
