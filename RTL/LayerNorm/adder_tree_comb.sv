`timescale 1ns / 1ps

module adder_tree #(
    parameter NUM_INPUTS = 32,
    parameter DATA_WIDTH = 32
)(
    // 32 parallel inputs coming directly from the Processing Elements
    input  logic signed [DATA_WIDTH-1:0] pe_data_in [0:NUM_INPUTS-1],
    
    // The single squashed output going to the Accumulator
    output logic signed [DATA_WIDTH-1:0] tree_sum_out
);

    // --------------------------------------------------------
    // Internal Wires for the Reduction Stages
    // --------------------------------------------------------
    logic signed [DATA_WIDTH-1:0] stage1 [0:15]; // 16 sums
    logic signed [DATA_WIDTH-1:0] stage2 [0:7];  // 8 sums
    logic signed [DATA_WIDTH-1:0] stage3 [0:3];  // 4 sums
    logic signed [DATA_WIDTH-1:0] stage4 [0:1];  // 2 sums

    always_comb begin
        // --------------------------------------------------------
        // Stage 1: 32 inputs -> 16 sums
        // --------------------------------------------------------
        for (int i = 0; i < 16; i++) begin
            stage1[i] = pe_data_in[2*i] + pe_data_in[2*i + 1];
        end

        // --------------------------------------------------------
        // Stage 2: 16 inputs -> 8 sums
        // --------------------------------------------------------
        for (int i = 0; i < 8; i++) begin
            stage2[i] = stage1[2*i] + stage1[2*i + 1];
        end

        // --------------------------------------------------------
        // Stage 3: 8 inputs -> 4 sums
        // --------------------------------------------------------
        for (int i = 0; i < 4; i++) begin
            stage3[i] = stage2[2*i] + stage2[2*i + 1];
        end

        // --------------------------------------------------------
        // Stage 4: 4 inputs -> 2 sums
        // --------------------------------------------------------
        for (int i = 0; i < 2; i++) begin
            stage4[i] = stage3[2*i] + stage3[2*i + 1];
        end

        // --------------------------------------------------------
        // Stage 5: 2 inputs -> Final 1 sum
        // --------------------------------------------------------
        tree_sum_out = stage4[0] + stage4[1];
    end

endmodule