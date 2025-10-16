`timescale 1ns/1ps

module PolynomialUnit_tb;

  parameter int Q = 26;
  parameter int W = 32;
  parameter real Q_SCALE = 2.0**Q;
  
  // Clock and reset
  logic clk;
  logic rst_n;
  
  // DUT signals
  logic valid_in;
  logic signed [W-1:0] xi_q;
  logic valid_out;
  logic signed [W-1:0] s_xi_q;
  
  // Test data
  real test_inputs[10] = '{-2.0, -1.5, -1.0, -0.5, 0.0, 0.5, 1.0, 1.5, 2.0, 2.5};
  int input_idx = 0;
  int output_idx = 0;
  
  // Instantiate DUT
  PolynomialUnit #(
    .Q(Q),
    .W(W)
  ) dut (
    .clk(clk),
    .rst_n(rst_n),
    .valid_in(valid_in),
    .xi_q(xi_q),
    .valid_out(valid_out),
    .s_xi_q(s_xi_q)
  );
  
  // Clock generation (10ns period = 100MHz)
  initial begin
    clk = 0;
    forever #5 clk = ~clk;
  end
  
  // Input stimulus - Send one input EVERY cycle
  initial begin
    $display("=== GELU Polynomial Streaming Test ===");
    $display("Sending one input per cycle (fully pipelined)");
    $display("Latency: 7 cycles, Throughput: 1 result/cycle\n");
    
    // Initialize
    rst_n = 0;
    valid_in = 0;
    xi_q = 0;
    
    // Reset
    repeat(3) @(posedge clk);
    rst_n = 1;
    repeat(2) @(posedge clk);
    
    // Stream inputs continuously
    for (input_idx = 0; input_idx < 10; input_idx++) begin
      @(posedge clk);
      valid_in = 1'b1;
      xi_q = $rtoi(test_inputs[input_idx] * Q_SCALE);
      $display("[Cycle %2d] INPUT:  xi = %7.3f (0x%08h)", 
               $time/10, test_inputs[input_idx], xi_q);
    end
    
    // Stop sending inputs
    @(posedge clk);
    valid_in = 1'b0;
    xi_q = 0;
    
    // Wait for all outputs
    repeat(20) @(posedge clk);
    
    $display("\n=== Test Complete ===");
    $display("Total inputs sent: 10");
    $display("Total outputs received: %0d", output_idx);
    $finish;
  end
  
  // Output monitor - Capture outputs as they arrive
  always @(posedge clk) begin
    if (valid_out) begin
      real s_actual, s_expected, xi_val, error;
      
      xi_val = test_inputs[output_idx];
      s_expected = -2.30220819814 * (xi_val + 0.044715 * xi_val * xi_val * xi_val);
      s_actual = real'($signed(s_xi_q)) / Q_SCALE;
      error = s_actual - s_expected;
      
      $display("[Cycle %2d] OUTPUT: s(xi=%7.3f) = %10.6f | Expected: %10.6f | Error: %9.6f", 
               $time/10, xi_val, s_actual, s_expected, error);
      
      output_idx++;
    end
  end
  
  // Performance monitor
  initial begin
    static int first_output_cycle = -1;
    static int last_output_cycle = -1;

    // Detect first output
    @(posedge valid_out);
    first_output_cycle = $time / 10;
    $display("\n*** First output at cycle %0d (Latency = %0d cycles) ***", 
             first_output_cycle, first_output_cycle - 5);
    
    // Detect last output
    fork
      begin
        repeat(9) @(posedge clk iff valid_out);
        last_output_cycle = $time / 10;
      end
    join_none
    
    // Calculate throughput
    #500;
    if (last_output_cycle > 0) begin
      static int total_cycles = last_output_cycle - first_output_cycle + 1;
      static real throughput_cycles = real'(total_cycles) / 10.0;
      $display("\n*** Performance Summary ***");
      $display("Latency: %0d cycles", first_output_cycle - 5);
      $display("Throughput: 1 result every %.1f cycles (%.1f results/10 cycles)",
               throughput_cycles, 10.0/throughput_cycles);
      if (throughput_cycles <= 1.1)
        $display("✓ Achieving target: 1 result per cycle!");
    end
  end
  
  // Timeout
  initial begin
    #2000;
    $display("ERROR: Testbench timeout!");
    $finish;
  end

endmodule
