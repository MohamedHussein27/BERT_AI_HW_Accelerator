module tb_integrated_tree_acc;

    // --------------------------------------------------------
    // Parameters
    // --------------------------------------------------------
    localparam int NUM_INPUTS = 32;
    localparam int DATAWIDTH = 32;       // Q5.26
    localparam int DATAWIDTH_OUTPUT = 36; // Q9.26
    localparam real Q_SCALE = 2.0**26;    // Scale factor for QX.26 format

    // --------------------------------------------------------
    // Signals
    // --------------------------------------------------------
    logic clk;
    logic rst_n;
    logic valid_in;
    logic fetch;

    // Adder Tree Interface
    logic signed [DATAWIDTH-1:0] pe_data_in [0:NUM_INPUTS-1];
    logic signed [DATAWIDTH-1:0] tree_sum_out;

    // Accumulator Output
    logic signed [DATAWIDTH_OUTPUT-1:0] data_out;

    // --------------------------------------------------------
    // DUT Instantiations
    // --------------------------------------------------------
    
    // 1. Instantiate the Adder Tree
    adder_tree #(
        .NUM_INPUTS(NUM_INPUTS),
        .DATA_WIDTH(DATAWIDTH)
    ) dut_tree (
        .pe_data_in(pe_data_in),
        .tree_sum_out(tree_sum_out)
    );

    // 2. Instantiate the Accumulator (Input driven by Adder Tree)
    accumulator #(
        .DATAWIDTH(DATAWIDTH),
        .DATAWIDTH_OUTPUT(DATAWIDTH_OUTPUT)
    ) dut_acc (
        .clk(clk),
        .rst_n(rst_n),
        .valid_in(valid_in),
        .fetch(fetch),
        .data_in(tree_sum_out), // <-- Connected directly to Adder Tree output
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
        int  chunk_sum_golden;     // 32-bit variable to verify the adder tree
        longint expected_full_sum; // 64-bit variable for the final expected result
        
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
        #10;

        // Generate 768 elements, processed in 24 cycles (24 * 32 = 768)
        for (int i = 0; i < 24; i++) begin
            chunk_sum_golden = 0; 
            
            // 1. Populate the 32 parallel inputs for the Adder Tree
            for (int j = 0; j < 32; j++) begin
                // Generate random float between -8.33 and +8.33
                rand_val = (real'($random % 1000) / 120.0);
                
                // Convert float to 32-bit fixed point Q5.26
                val_fixed = int'(rand_val * Q_SCALE);
                
                // Assign to the hardware adder tree input array
                pe_data_in[j] = val_fixed;
                
                // Calculate the expected sum in software for our golden reference check
                chunk_sum_golden += val_fixed; 
            end

            // Wait a brief delta time to allow combinational logic in the adder tree to evaluate
            #1; 

            // 2. Drive the valid signal to register the tree_sum_out into the Accumulator
            @(negedge clk);
            valid_in <= 1'b1;

            // Update our golden expected sum (sign-extending the 32-bit expected chunk)
            expected_full_sum += longint'(chunk_sum_golden);
        end

        // 3. Stop feeding data and fetch the final accumulated result
        @(negedge clk);
        valid_in <= 1'b0;
        @(posedge clk);
        fetch    <= 1'b1;

        // Drop fetch on the next cycle
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