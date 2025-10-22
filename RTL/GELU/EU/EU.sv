`timescale 1ns/1ps

module EU #(
  parameter int Q = 26,           // fractional bits (Q5.26)
  parameter int W = 32,           // data width (bits)
  parameter int INT_WIDTH = 5     // integer part width
) (
  input  logic                     clk,
  input  logic                     rst_n,
  input  logic                     valid_in,
  
  // Inputs from Integer-Fractional Separator
  input  logic [INT_WIDTH-1:0]     integer_part,   // full integer part
  input  logic [Q-1:0]             frac_part,      // fractional part
  
  // LUT outputs (these would come from a separate LUT module)
  input  logic signed [W-1:0]      k_coeff,        // slope coefficient from LUT
  input  logic signed [W-1:0]      b_intercept,    // intercept from LUT
  
  output logic                     valid_out,
  output logic signed [W-1:0]      exp_result      // 2^x approximation result
);

  // Pipeline Stage 1: Multiply fractional part by k
  logic signed [63:0]              k_frac_product_s1;
  logic signed [W-1:0]             b_intercept_s1;
  logic [INT_WIDTH-1:0]            integer_s1;
  logic                            valid_s1;
  
  // Pipeline Stage 2: Scale product and add intercept
  logic signed [W-1:0]             linear_interp_s2;
  logic [INT_WIDTH-1:0]            integer_s2;
  logic                            valid_s2;
  
  // Pipeline Stage 3: Shift interpolation result by integer part
  logic signed [W-1:0]             final_result_s3;
  logic                            valid_s3;
  
  always_ff @(posedge clk or negedge rst_n) begin
    if (!rst_n) begin
      // Stage 1
      k_frac_product_s1 <= 64'sd0;
      b_intercept_s1    <= '0;
      integer_s1        <= '0;
      valid_s1          <= 1'b0;
      
      // Stage 2
      linear_interp_s2  <= '0;
      integer_s2        <= '0;
      valid_s2          <= 1'b0;
      
      // Stage 3
      final_result_s3   <= '0;
      valid_s3          <= 1'b0;
      
    end else begin
      
      // Stage 1: Multiply Xfrac × k
      valid_s1 <= valid_in;
      
      if (valid_in) begin        
        // Multiply: frac_part × k_coeff
        k_frac_product_s1 <= $signed({{INT_WIDTH{1'b0}}, frac_part}) * $signed(k_coeff);
        
        // Pass through intercept and integer
        b_intercept_s1    <= b_intercept;
        integer_s1        <= integer_part;
      end
      
      // Stage 2: Scale and add intercept (frac × k + b)
      valid_s2 <= valid_s1;
      
      if (valid_s1) begin
        // Scale the product back to Q format and add intercept
        linear_interp_s2 <= ($signed(k_frac_product_s1) >>> Q) + $signed(b_intercept_s1);
        
        // Pass through integer part
        integer_s2 <= integer_s1;
      end
      
      // Stage 3: Shift interpolation result by integer_part
      valid_s3 <= valid_s2;
      
      if (valid_s2) begin
        automatic int shift_amt;
        automatic int abs_shift;
        
        // Use integer_part directly as shift amount (sign-extended to 32 bits)
        shift_amt = $signed({{(32-INT_WIDTH){integer_s2[INT_WIDTH-1]}}, integer_s2});
        
        // Shift the interpolation result by the integer part
        // Positive shift_amt = left shift (multiply by 2^integer_part)
        // Negative shift_amt = right shift (divide by 2^integer_part)
        if (shift_amt >= 0) begin
          if (shift_amt >= W) begin
            // Saturation based on sign of interpolation result
            if (linear_interp_s2[W-1] == 1'b0)
              final_result_s3 <= {1'b0, {(W-1){1'b1}}};  // Max positive
            else
              final_result_s3 <= {W{1'b1}};              // All ones for negative
          end else begin
            // Arithmetic left shift
            final_result_s3 <= $signed(linear_interp_s2) <<< shift_amt;
          end
        end else begin
          // Negative shift amount - right shift
          abs_shift = -shift_amt;
          if (abs_shift >= W) begin
            // Sign extension for large right shifts
            final_result_s3 <= {W{linear_interp_s2[W-1]}};
          end else begin
            // Arithmetic right shift
            final_result_s3 <= $signed(linear_interp_s2) >>> abs_shift;
          end
        end
      end
    end
  end
  
  // Output assignment
  assign exp_result = final_result_s3;
  assign valid_out  = valid_s3;

endmodule
