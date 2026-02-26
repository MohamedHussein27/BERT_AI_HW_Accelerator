library verilog;
use verilog.vl_types.all;
entity integration_debug_tb is
    generic(
        VEC_LEN         : integer := 4;
        CLK_PERIOD      : integer := 10
    );
    attribute mti_svvh_generic_type : integer;
    attribute mti_svvh_generic_type of VEC_LEN : constant is 2;
    attribute mti_svvh_generic_type of CLK_PERIOD : constant is 2;
end integration_debug_tb;
