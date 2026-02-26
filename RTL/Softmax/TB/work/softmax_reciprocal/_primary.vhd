library verilog;
use verilog.vl_types.all;
entity softmax_reciprocal is
    generic(
        W               : integer := 32;
        Q               : integer := 24;
        LUT_BITS        : integer := 4;
        ITER            : integer := 2
    );
    port(
        clk             : in     vl_logic;
        rst_n           : in     vl_logic;
        in_valid        : in     vl_logic;
        in_data         : in     vl_logic_vector;
        out_valid       : out    vl_logic;
        out_data        : out    vl_logic_vector
    );
    attribute mti_svvh_generic_type : integer;
    attribute mti_svvh_generic_type of W : constant is 2;
    attribute mti_svvh_generic_type of Q : constant is 2;
    attribute mti_svvh_generic_type of LUT_BITS : constant is 2;
    attribute mti_svvh_generic_type of ITER : constant is 2;
end softmax_reciprocal;
