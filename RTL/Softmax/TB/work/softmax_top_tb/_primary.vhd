library verilog;
use verilog.vl_types.all;
entity softmax_top_tb is
    generic(
        VEC_LEN         : integer := 64;
        NUM_VECTORS     : integer := 10;
        TOTAL_ELEMS     : vl_logic_vector(31 downto 0);
        CLK_PERIOD      : integer := 10;
        TIMEOUT         : integer := 500000
    );
    attribute mti_svvh_generic_type : integer;
    attribute mti_svvh_generic_type of VEC_LEN : constant is 2;
    attribute mti_svvh_generic_type of NUM_VECTORS : constant is 2;
    attribute mti_svvh_generic_type of TOTAL_ELEMS : constant is 4;
    attribute mti_svvh_generic_type of CLK_PERIOD : constant is 2;
    attribute mti_svvh_generic_type of TIMEOUT : constant is 2;
end softmax_top_tb;
