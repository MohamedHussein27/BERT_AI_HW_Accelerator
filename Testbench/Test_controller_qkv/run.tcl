# ============================================================================
# ModelSim/QuestaSim DO file for QKV Integration Testbench
# ============================================================================

# 1. Create and map the work library
vlib work
vmap work work

# 2. Compile the design files
# Note: Using -sv to enable SystemVerilog features for both .v and .sv files
vlog *.*v
# 3. Load the simulation
# -voptargs=+acc ensures that signals are not optimized away so we can view them in the wave window
vsim -voptargs=+acc work.tb_integration_qkv

# 4. Configure Waveform Viewer
# Clear existing waves
delete wave *

# Add Clock, Reset, and Top-level Control
add wave -divider "System Signals"
add wave -color "Yellow" sim:/tb_integration_qkv/clk
add wave -color "Yellow" sim:/tb_integration_qkv/rst_n
add wave sim:/tb_integration_qkv/start_inference
add wave sim:/tb_integration_qkv/layer_done_wire

# Add Controller State
add wave -divider "Master Controller"
add wave -radix unsigned sim:/tb_integration_qkv/u_cu/state
add wave -position insertpoint  \
sim:/tb_integration_qkv/u_cu/sa_first_iter \
sim:/tb_integration_qkv/u_cu/sa_last_tile \
sim:/tb_integration_qkv/u_cu/done_counter \
sim:/tb_integration_qkv/u_cu/Q_K_V_sel \
sim:/tb_integration_qkv/u_cu/tile_done_counter \
sim:/tb_integration_qkv/u_cu/fetch_double_buf \
sim:/tb_integration_qkv/u_cu/sa_first_iter_counter

# Add Fetch Logic Signals
add wave -divider "Fetch Logic"
add wave sim:/tb_integration_qkv/fetch_start_wire
add wave -radix binary sim:/tb_integration_qkv/fetch_buffer_sel_wire
add wave sim:/tb_integration_qkv/fetch_busy_wire
add wave sim:/tb_integration_qkv/fetch_wt_done_wire
add wave sim:/tb_integration_qkv/fetch_in_done_wire
add wave -radix unsigned sim:/tb_integration_qkv/fetch_bram_addr_wire

# Add Write Logic Signals
add wave -divider "Write Logic"
add wave sim:/tb_integration_qkv/write_start_wire
add wave -radix binary sim:/tb_integration_qkv/write_buffer_sel_wire
add wave sim:/tb_integration_qkv/write_busy_wire
add wave sim:/tb_integration_qkv/write_done_all_wire
add wave -radix unsigned sim:/tb_integration_qkv/write_bram_addr_wire
add wave sim:/tb_integration_qkv/u_write_logic/write_offset 
add wave sim:/tb_integration_qkv/u_write_logic/reset_addr_counter
add wave sim:/tb_integration_qkv/u_write_logic/WRITE_START_OFFSET
add wave sim:/tb_integration_qkv/u_write_logic/current_state 
add wave -position insertpoint  \
sim:/tb_integration_qkv/u_qkv_buffer/wea \
sim:/tb_integration_qkv/u_qkv_buffer/dina \
sim:/tb_integration_qkv/sa_data_out_wire \
sim:/tb_integration_qkv/vq_data_out_wire \
sim:/tb_integration_qkv/vq_valid_out_wire \

# Add Systolic Array Control Signals
add wave -divider "Systolic Array "
add wave -position insertpoint  \
sim:/tb_integration_qkv/u_systolic/valid_in \
sim:/tb_integration_qkv/u_systolic/in_A \
sim:/tb_integration_qkv/u_systolic/in_A_wire \
sim:/tb_integration_qkv/u_systolic/in_B_wire \
sim:/tb_integration_qkv/u_systolic/interbuffer_output \
sim:/tb_integration_qkv/u_systolic/rd_addr \
sim:/tb_integration_qkv/u_systolic/interbuffer_intput \
sim:/tb_integration_qkv/u_systolic/out_C_wire \
sim:/tb_integration_qkv/u_systolic/we \
sim:/tb_integration_qkv/u_systolic/contoller/cs \
sim:/tb_integration_qkv/u_systolic/contoller/cycle_cnt \
sim:/tb_integration_qkv/u_systolic/first_iteration \
sim:/tb_integration_qkv/u_systolic/zero_in \
sim:/tb_integration_qkv/u_systolic/load_weight \
sim:/tb_integration_qkv/u_systolic/last_tile \
sim:/tb_integration_qkv/u_systolic/busy \
sim:/tb_integration_qkv/u_systolic/done \
sim:/tb_integration_qkv/u_wbi_buffer/doutb \
sim:/tb_integration_qkv/u_systolic/valid_out \
sim:/tb_integration_qkv/u_systolic/pre_valid_out
# 5. Run the simulation
# The testbench will halt automatically when it hits $finish;
run -all

# Zoom full to see the whole simulation run
wave zoom full



