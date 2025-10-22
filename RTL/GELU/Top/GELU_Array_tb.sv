`timescale 1ns/1ps
module GELU_Array_tb;
  // Parameters
  localparam int Q          = 26;
  localparam int W          = 32;
  localparam int INT_WIDTH  = 5;
  localparam int NUM_LANES  = 32;
  // DUT I/O
  logic                            clk;
  logic                            rst_n;
  logic                            valid_in;
  logic signed [W-1:0]             xi [NUM_LANES-1:0];
  logic                            valid_out;
  logic signed [W-1:0]             gelu_out [NUM_LANES-1:0];
  // Clock
  initial clk = 0;
  always #5 clk = ~clk;
  // DUT
  GELU_Array #(
    .Q(Q),
    .W(W),
    .INT_WIDTH(INT_WIDTH),
    .NUM_LANES(NUM_LANES)
  ) dut (
    .clk(clk),
    .rst_n(rst_n),
    .valid_in(valid_in),
    .xi(xi),
    .valid_out(valid_out),
    .gelu_out(gelu_out)
  );
  // Reset task
  task reset_dut;
    integer i;
    begin
      rst_n = 0;
      valid_in = 0;
      for (i = 0; i < NUM_LANES; i = i + 1)
        xi[i] = '0;
      repeat (5) @(posedge clk);
      rst_n = 1;
      $display("[%0t] DUT Reset Completed.", $time);
    end
  endtask
  // Reference GELU (float version)
  function real gelu_ref;
    input real x;
    real pi, sqrt2pi;
    begin
      pi = 3.14159265358979;
      sqrt2pi = sqrt(2.0 / pi);
      gelu_ref = 0.5 * x * (1.0 + tanh(sqrt2pi * (x + 0.044715 * x * x * x)));
    end
  endfunction
  // Conversion helpers
  function real fixed_to_real;
    input signed [W-1:0] val;
    begin
      fixed_to_real = val / $itor(1 << Q);
    end
  endfunction

  function signed [W-1:0] real_to_fixed;
    input real val;
    begin
      real_to_fixed = $rtoi(val * (1 << Q));
    end
  endfunction
  // Percent error
  function real percent_error;
    input real ref_val;
    input real dut_val;
    real diff;
    begin
      diff = dut_val - ref_val;
      if (ref_val == 0.0)
        percent_error = fabs(diff) * 100.0;
      else
        percent_error = fabs(diff / ref_val) * 100.0;
    end
  endfunction
  // Main stimulus
  integer errors;
  integer t, i;
  real x_real, y_ref, y_dut;
  real abs_err, perc_err, total_perc_err;

  initial begin
    errors = 0;
    total_perc_err = 0.0;

    $display("\n==============================================================");
    $display("               GELU_Array Testbench with Accuracy");
    $display("==============================================================\n");

    reset_dut();
    repeat (3) @(posedge clk);

    $display("[%0t] Starting accuracy test...", $time);

    for (t = -30; t <= 30; t = t + 2) begin
      valid_in = 1;
      for (i = 0; i < NUM_LANES; i = i + 1) begin
        x_real = t + (i * 0.1);
        xi[i] = real_to_fixed(x_real);
      end
      @(posedge clk);
      valid_in = 0;

      wait (valid_out);
      @(posedge clk);

      for (i = 0; i < NUM_LANES; i = i + 1) begin
        x_real = fixed_to_real(xi[i]);
        y_ref = gelu_ref(x_real);
        y_dut = fixed_to_real(gelu_out[i]);
        abs_err = fabs(y_dut - y_ref);
        perc_err = percent_error(y_ref, y_dut);
        total_perc_err = total_perc_err + perc_err;

        if (perc_err > 5.0) begin
          $display("[WARN] Lane %0d | x=%6.3f | Ref=%8.6f | DUT=%8.6f | Err=%6.2f%%",
                   i, x_real, y_ref, y_dut, perc_err);
          errors = errors + 1;
        end
      end
    end

    total_perc_err = total_perc_err / (NUM_LANES * 31.0);

    $display("\nAverage %% Error across all lanes: %.3f%%", total_perc_err);
    if (errors == 0)
      $display("  PASS: All outputs within tolerance.\n");
    else
      $display("  FAIL: %0d outputs exceeded 5%% error tolerance.\n", errors);

    $display("==============================================================");
    $display("                Simulation completed.");
    $display("==============================================================");
    $finish;
  end

endmodule
