# ==============================================================================
# ModelSim/QuestaSim DO file for Systolic Array Testbench
# ==============================================================================

# Close any open simulation
quit -sim

# Create/clean work library
vlib work
vmap work work

# Compile all source files
echo "Compiling RTL files..."
vlog -sv +incdir+. PE.sv
vlog -sv +incdir+. skew_buffer.sv
vlog -sv +incdir+. systolic_internal_buffer.sv
vlog -sv +incdir+. systolic.sv
vlog -sv +incdir+. systolic_controller.sv
vlog -sv +incdir+. systolic_top.sv

echo "Compiling testbench..."
vlog -sv +incdir+. systolic_tb_st2.sv

# Start simulation
echo "Starting simulation..."
vsim -voptargs=+acc work.systolic_tb_st2

add wave -position insertpoint sim:/systolic_tb_st2/*
add wave -position insertpoint sim:/systolic_tb_st2/dut/contoller/*
add wave -position insertpoint sim:/systolic_tb_st2/dut/u_systolic/*
add wave -position insertpoint sim:/systolic_tb_st2/dut/partial_sum_buffer/*

# Run simulation
echo "Running simulation..."
run 20ms

# Zoom to fit
wave zoom full
