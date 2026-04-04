vlib work
vlog PE_pkg.sv accumulator.sv adder_tree_comb.sv CU.sv inv_sqrt_lut.sv inv_sqrt.sv PE.sv  LayerNorm_top.sv LayerNorm_Top_tb.sv
vsim -voptargs=+acc work.tb_layernorm_top
add wave -position insertpoint  \
{sim:/tb_layernorm_top/dut/gen_pes[0]/u_pe/local_mean_reg}
add wave -position insertpoint  \
sim:/tb_layernorm_top/dut/u_fsm/ROW_CNT_WIDTH \
sim:/tb_layernorm_top/dut/u_fsm/clk \
sim:/tb_layernorm_top/dut/u_fsm/rst_n \
sim:/tb_layernorm_top/dut/u_fsm/data_valid \
sim:/tb_layernorm_top/dut/u_fsm/pe_opcode \
sim:/tb_layernorm_top/dut/u_fsm/accum_en \
sim:/tb_layernorm_top/dut/u_fsm/accum_fetch \
sim:/tb_layernorm_top/dut/u_fsm/sqrt_valid_in \
sim:/tb_layernorm_top/dut/u_fsm/sqrt_valid_out \
sim:/tb_layernorm_top/dut/u_fsm/sqrt_busy \
sim:/tb_layernorm_top/dut/u_fsm/out_valid \
sim:/tb_layernorm_top/dut/u_fsm/done \
sim:/tb_layernorm_top/dut/u_fsm/state \
sim:/tb_layernorm_top/dut/u_fsm/next_state \
sim:/tb_layernorm_top/dut/u_fsm/load_parameters \
sim:/tb_layernorm_top/dut/u_fsm/mean_var \
sim:/tb_layernorm_top/dut/u_fsm/chunk_cnt \
sim:/tb_layernorm_top/dut/u_fsm/row_cnt
add wave -position insertpoint  \
sim:/tb_layernorm_top/dut/u_accum/DATAWIDTH \
sim:/tb_layernorm_top/dut/u_accum/DATAWIDTH_OUTPUT \
sim:/tb_layernorm_top/dut/u_accum/clk \
sim:/tb_layernorm_top/dut/u_accum/rst_n \
sim:/tb_layernorm_top/dut/u_accum/valid_in \
sim:/tb_layernorm_top/dut/u_accum/fetch \
sim:/tb_layernorm_top/dut/u_accum/data_in \
sim:/tb_layernorm_top/dut/u_accum/data_out \
sim:/tb_layernorm_top/dut/u_accum/acc_reg
run -all