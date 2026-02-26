library verilog;
use verilog.vl_types.all;
entity exp_pla_tb is
    generic(
        NUM_TESTS       : integer := 200;
        CLK_PERIOD      : integer := 10
    );
    attribute mti_svvh_generic_type : integer;
    attribute mti_svvh_generic_type of NUM_TESTS : constant is 2;
    attribute mti_svvh_generic_type of CLK_PERIOD : constant is 2;
end exp_pla_tb;
