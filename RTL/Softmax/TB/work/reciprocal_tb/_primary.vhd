library verilog;
use verilog.vl_types.all;
entity reciprocal_tb is
    generic(
        NUM_TESTS       : integer := 50;
        CLK_PERIOD      : integer := 10
    );
    attribute mti_svvh_generic_type : integer;
    attribute mti_svvh_generic_type of NUM_TESTS : constant is 2;
    attribute mti_svvh_generic_type of CLK_PERIOD : constant is 2;
end reciprocal_tb;
