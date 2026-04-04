`timescale 1ns / 1ps

module adder_tree #(
    parameter NUM_INPUTS = 32,
    parameter DATA_WIDTH_IN = 37,
    parameter DATA_WIDTH_OUT = 42
)(
    input  logic signed [DATA_WIDTH_IN-1:0]  pe_data_in [0:NUM_INPUTS-1],
    output logic signed [DATA_WIDTH_OUT-1:0] tree_sum_out
);

    logic signed [DATA_WIDTH_IN:0]   stage1 [0:15]; // 38 bits
    logic signed [DATA_WIDTH_IN+1:0] stage2 [0:7];  // 39 bits
    logic signed [DATA_WIDTH_IN+2:0] stage3 [0:3];  // 40 bits
    logic signed [DATA_WIDTH_IN+3:0] stage4 [0:1];  // 41 bits


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