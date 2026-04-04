`timescale 1ns / 1ps
import PE_pkg::*;

module tb_layernorm_top;

    parameter DATAWIDTH = 32;
    localparam real Q_SCALE = 67108864.0; // 2^26 

    logic clk;
    logic rst_n;
    logic data_valid;
    logic signed [DATAWIDTH-1:0] buffer_rdata [0:31];
    logic signed [DATAWIDTH-1:0] norm_out_data [0:31];
    logic norm_out_valid;
    logic done, busy;

    layernorm_top #(
        .DATAWIDTH(DATAWIDTH)
    ) dut (
        .clk(clk),
        .rst_n(rst_n),
        .data_valid(data_valid),
        .buffer_rdata(buffer_rdata),
        .norm_out_data(norm_out_data),
        .norm_out_valid(norm_out_valid),
        .done(done),
        .busy(busy)
    );

    always #5 clk = ~clk;

    logic signed [DATAWIDTH-1:0] test_row [0:767]; 
    logic signed [DATAWIDTH-1:0] tb_gamma [0:31]; 
    logic signed [DATAWIDTH-1:0] tb_beta  [0:31]; 
    
    real expected_mean;
    real expected_var;
    real expected_inv_std;
    real expected_out [0:767];

    int out_chunk_idx = 0; 
    int errors = 0;

    initial begin
        int i;
        clk = 0;
        rst_n = 0;
        data_valid = 0;
        for (i=0; i<32; i++) buffer_rdata[i] = 0;
        
        $display("==================================================");
        $display("   STARTING CASCADED LAYERNORM VERIFICATION");
        $display("==================================================");

        generate_test_data();

        #20 rst_n = 1;
        #10;

        $display("\n[TIME %0t] Starting PASS 1 (Mean Computation)...", $time);
        stream_row();
        #20; 

        $display("[TIME %0t] Starting PASS 2 (Variance Computation)...", $time);
        stream_row();

        wait(dut.u_fsm.state == 3'd4); // ST_CALC_SQRT
        $display("[TIME %0t] FSM entered SQRT Wait. Injecting Gamma and Beta!", $time);
        
        @(negedge clk);
        data_valid = 1'b1; 
        for (i = 0; i < 32; i++) buffer_rdata[i] = tb_gamma[i];
        
        @(negedge clk);
        for (i = 0; i < 32; i++) buffer_rdata[i] = tb_beta[i];
        
        @(negedge clk);
        data_valid = 1'b0;
        for (i = 0; i < 32; i++) buffer_rdata[i] = '0;

        wait(dut.u_fsm.state == 3'd5); // ST_PASS3_NORM

        $display("[TIME %0t] Starting PASS 3 (Normalization + Affine)...", $time);
        stream_row();

        // Wait for background verification block to verify all chunks
        wait(out_chunk_idx == 23);
        $display("\n[TIME %0t] Hardware finished processing all data!", $time);

        $display("\n==================================================");
        if (errors == 0) $display("   SUCCESS! Hardware matches Golden Model perfectly.");
        else             $display("   FAILED! Detected %0d mismatches.", errors);
        $display("==================================================");
        #50;
        $finish;
    end

    task automatic stream_row();
        int chunk, i;
        begin
            for (chunk = 0; chunk < 24; chunk++) begin
                @(negedge clk);
                data_valid = 1'b1;
                for (i = 0; i < 32; i++) begin
                    buffer_rdata[i] = test_row[(chunk * 32) + i];
                end
            end
            @(negedge clk);
            data_valid = 1'b0; 
        end
    endtask

    always @(negedge clk) begin
        if (rst_n && norm_out_valid) begin
            compare();
        end
    end

    task compare (); 
        begin
            int i;
            real hw_real_out;
            real diff;
            int absolute_idx;
            
            for (i = 0; i < 32; i++) begin
                absolute_idx = (out_chunk_idx * 32) + i;
                
                hw_real_out = real'(norm_out_data[i]) / Q_SCALE;
                diff = hw_real_out - expected_out[absolute_idx];
                if (diff < 0) diff = -diff;

                if (diff > 0.05) begin
                    $error("Mismatch at Index %0d | HW: %f | Expected: %f", absolute_idx, hw_real_out, expected_out[absolute_idx]);
                    errors++;
                end
                
                if (absolute_idx < 3) begin
                    $display("   -> Index %0d | HW: %8f | Expected: %8f", absolute_idx, hw_real_out, expected_out[absolute_idx]);
                end
            end
            out_chunk_idx++;
        end
    endtask

    task automatic generate_test_data(); 
        int i;
        int pe_idx;
        real sum;
        real sqr_sum;
        real rand_val;
        real real_val;
        real diff;
        real real_gamma;
        real real_beta; 
        real norm_val;
        
        begin
            sum = 0.0;
            sqr_sum = 0.0;
            
            for (i = 0; i < 32; i++) begin
                tb_gamma[i] = int'((1.0 + (real'($random % 100) / 1000.0)) * Q_SCALE); 
                tb_beta[i]  = int'((0.1 + (real'($random % 100) / 1000.0)) * Q_SCALE); 
            end

            for (i = 0; i < 768; i++) begin
                rand_val = (real'($random % 1000) / 120.0); 
                test_row[i] = int'(rand_val * Q_SCALE);       
                sum += rand_val;
            end

            expected_mean = sum / 768.0;

            for (i = 0; i < 768; i++) begin
                real_val = real'(test_row[i]) / Q_SCALE;
                diff = real_val - expected_mean;
                sqr_sum += (diff * diff);
            end
            expected_var = sqr_sum / 768.0;

            expected_inv_std = 1.0 / $sqrt(expected_var + 0.00001);

            for (i = 0; i < 768; i++) begin
                pe_idx = i % 32; 
                
                real_val   = real'(test_row[i]) / Q_SCALE;
                real_gamma = real'(tb_gamma[pe_idx]) / Q_SCALE;
                real_beta  = real'(tb_beta[pe_idx]) / Q_SCALE;
                
                norm_val = (real_val - expected_mean) * expected_inv_std;
                expected_out[i] = (norm_val * real_gamma) + real_beta;
            end
            
            $display("   [GOLDEN] Expected Mean     : %f", expected_mean);
            $display("   [GOLDEN] Expected Variance : %f", expected_var);
            $display("   [GOLDEN] Expected 1/StdDev : %f\n", expected_inv_std);
        end
    endtask

endmodule