`timescale 1ns / 1ps

module tb_integrated_tree_acc;

    // --------------------------------------------------------
    // Parameters
    // --------------------------------------------------------
    localparam int NUM_INPUTS = 32;
    localparam int DATAWIDTH_IN = 32;       // Input Q5.26
    localparam int DATAWIDTH_TREE_OUT = 37; // Tree output width
    localparam int DATAWIDTH_OUTPUT = 42;   // Final Acc Output Q15.26
    localparam real Q_SCALE = 2.0**26;      // Scale factor for QX.26 format

    // --------------------------------------------------------
    // Signals
    // --------------------------------------------------------
    logic clk;
    logic rst_n;
    logic valid_in;
    logic fetch;

    // Adder Tree Interface
    logic signed [DATAWIDTH_IN-1:0] pe_data_in [0:NUM_INPUTS-1];
    logic signed [DATAWIDTH_TREE_OUT-1:0] tree_sum_out;

    // Accumulator Output
    logic signed [DATAWIDTH_OUTPUT-1:0] data_out;

    // --------------------------------------------------------
    // DUT Instantiations
    // --------------------------------------------------------
    
    // 1. Instantiate the Adder Tree
    adder_tree #(
        .NUM_INPUTS(NUM_INPUTS),
        .DATA_WIDTH_IN(DATAWIDTH_IN),
        .DATA_WIDTH_OUT(DATAWIDTH_TREE_OUT)
    ) dut_tree (
        .pe_data_in(pe_data_in),
        .tree_sum_out(tree_sum_out)
    );

    // 2. Instantiate the Accumulator (Input driven by Adder Tree)
    accumulator #(
        .DATAWIDTH_IN(DATAWIDTH_TREE_OUT),
        .DATAWIDTH_OUTPUT(DATAWIDTH_OUTPUT)
    ) dut_acc (
        .clk(clk),
        .rst_n(rst_n),
        .valid_in(valid_in),
        .fetch(fetch),
        .data_in(tree_sum_out),
        .data_out(data_out)
    );

    // --------------------------------------------------------
    // Clock Generation (100MHz)
    // --------------------------------------------------------
    initial begin
        clk = 0;
        forever #5 clk = ~clk;
    end

    // --------------------------------------------------------
    // Stimulus and Verification
    // --------------------------------------------------------
    initial begin
        real rand_val;
        int  val_fixed;
        longint chunk_sum_golden;     // Changed to 64-bit to prevent software overflow
        longint expected_full_sum;    // 64-bit variable for the final expected result
        
        // Initial values
        rst_n = 0;
        valid_in = 0;
        fetch = 0;
        expected_full_sum = 0;

        for (int i = 0; i < NUM_INPUTS; i++) begin
            pe_data_in[i] = '0;
        end

// Apply Reset
        #20;
        rst_n = 1;
        
        // ALIGN to a negative edge before we start driving data
        @(negedge clk); 

        // Generate 768 elements, processed in 24 cycles (24 * 32 = 768)
        for (int i = 0; i < 24; i++) begin
            chunk_sum_golden = 0;
            
            // 1. Populate the 32 parallel inputs for the Adder Tree
            for (int j = 0; j < 32; j++) begin
                rand_val = (real'($random % 1000) / 120.0);
                val_fixed = int'(rand_val * Q_SCALE);
                pe_data_in[j] = val_fixed;
                chunk_sum_golden += longint'(val_fixed);
            end

            // 2. Drive the valid signal to register the tree_sum_out into the Accumulator
            valid_in <= 1'b1;
            expected_full_sum += chunk_sum_golden;

            // 3. WAIT for the next negative edge. 
            // This ensures the positive edge safely samples the current chunk!
            @(negedge clk); 
        end

        // 3. Stop feeding data 
        valid_in <= 1'b0;
        
        // 4. Fetch the final accumulated result
        @(posedge clk);
        fetch    <= 1'b1;
        @(posedge clk);
        fetch    <= 1'b0;

        // Wait one cycle for the accumulator's register to output the data
        @(posedge clk);
        
        // 4. Verification and Console Output
        $display("--------------------------------------------------");
        $display("     Adder Tree + Accumulator Integration Test    ");
        $display("--------------------------------------------------");
        $display("Expected Sum (Hex)  : 0x%0h", expected_full_sum[DATAWIDTH_OUTPUT-1:0]);
        $display("Actual Sum (Hex)    : 0x%0h", data_out);
        $display("");
        $display("Expected Sum (Real) : %f", real'(expected_full_sum) / Q_SCALE);
        $display("Actual Sum (Real)   : %f", real'(data_out) / Q_SCALE);
        $display("--------------------------------------------------");
        
        if (data_out === expected_full_sum[DATAWIDTH_OUTPUT-1:0]) begin
            $display(">> STATUS: PASSED!");
        end else begin
            $display(">> STATUS: FAILED!");
        end
        $display("--------------------------------------------------");

        // End simulation
        #50 $finish;
    end
endmodule