module mul_pipelined #(
  parameter integer AW = 32,                // width of a_in
  parameter integer BW = 32,                // width of b_in
  parameter integer PIPELINE_STAGES = 1     // number of register stages after multiply (>=0)
) (
  input  wire                     clk,
  input  wire                     rst_n,   // active-low reset
  input  wire                     start,   // 1-cycle start pulse
  input  wire signed [AW-1:0]     a_in,
  input  wire signed [BW-1:0]     b_in,
  output reg                      busy,
  output reg                      done,    // 1-cycle pulse when p_out valid
  output reg signed [AW+BW-1:0]   p_out
);

  // local widths
  localparam integer OUTW = AW + BW;

  // Registers to hold operands (latched on start)
  reg signed [AW-1:0] a_reg;
  reg signed [BW-1:0] b_reg;

  // product register array for pipeline stages (stage 0 holds immediate product)
  reg signed [OUTW-1:0] prod_stage [0:PIPELINE_STAGES]; // prod_stage[0] is product after multiply

  // control: remaining cycles counter (optional)
  integer latency;
  integer counter;

  integer i;

  // Calculate latency: 1 cycle to capture & compute (we compute product in same cycle as capture),
  // plus PIPELINE_STAGES cycles of registers, then 1 cycle to present output.
  // We will implement such that done pulses when the last pipeline stage is ready.
  // Effective cycles from asserting start to done = 1 + PIPELINE_STAGES.
  initial begin
    latency = 1 + PIPELINE_STAGES;
  end

  // default reset
  always @(posedge clk or negedge rst_n) begin
    if (!rst_n) begin
      busy    <= 1'b0;
      done    <= 1'b0;
      p_out   <= {OUTW{1'b0}};
      a_reg   <= {AW{1'b0}};
      b_reg   <= {BW{1'b0}};
      for (i = 0; i <= PIPELINE_STAGES; i = i + 1) prod_stage[i] <= {OUTW{1'b0}};
      counter <= 0;
    end else begin
      done <= 1'b0; // default: done is single-cycle pulse

      // Start accepted only when not busy
      if (start && !busy) begin
        // latch operands
        a_reg <= a_in;
        b_reg <= b_in;
        // compute immediate product into stage 0 (combinational multiply assigned into reg)
        prod_stage[0] <= $signed(a_in) * $signed(b_in); // 32x32 -> 64 etc.
        // clear remaining stages (optional to avoid x)
        for (i = 1; i <= PIPELINE_STAGES; i = i + 1) prod_stage[i] <= {OUTW{1'b0}};
        // set busy & counter
        busy <= 1'b1;
        counter <= 1; // we've completed stage 0 in this cycle; need PIPELINE_STAGES more cycles
        if (PIPELINE_STAGES == 0) begin
          // when no extra pipeline, product is ready this cycle -> present next cycle
          // we'll present in the next clock (so latency = 1)
        end
      end else if (busy) begin
        // shift pipeline: move prod_stage[k-1] -> prod_stage[k]
        prod_stage[0] <= prod_stage[0]; // keep stage0 (no new multiplication unless new start accepted)
        for (i = 1; i <= PIPELINE_STAGES; i = i + 1) begin
          prod_stage[i] <= prod_stage[i-1];
        end

        // advance counter
        if (counter < latency) begin
          counter <= counter + 1;
        end

        // when counter reaches latency, output is valid this cycle
        if (counter >= latency) begin
          p_out <= prod_stage[PIPELINE_STAGES];
          done  <= 1'b1;
          busy  <= 1'b0;
          counter <= 0;
          // Optionally clear pipeline regs (not necessary)
        end
      end
    end
  end

endmodule
