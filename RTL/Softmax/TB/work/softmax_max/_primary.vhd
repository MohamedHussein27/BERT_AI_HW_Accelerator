library verilog;
use verilog.vl_types.all;
entity softmax_max is
    generic(
        VEC_LEN         : integer := 64;
        D_W             : integer := 32;
        MAX_LEN         : integer := 512;
        IDX_W           : integer := 10
    );
    port(
        clk             : in     vl_logic;
        rst_n           : in     vl_logic;
        start           : in     vl_logic;
        vec_len_cfg     : in     vl_logic_vector;
        in_valid        : in     vl_logic;
        in_ready        : out    vl_logic;
        in_data         : in     vl_logic_vector;
        max_valid       : out    vl_logic;
        max_value       : out    vl_logic_vector;
        rd_addr         : in     vl_logic_vector;
        rd_data         : out    vl_logic_vector
    );
    attribute mti_svvh_generic_type : integer;
    attribute mti_svvh_generic_type of VEC_LEN : constant is 2;
    attribute mti_svvh_generic_type of D_W : constant is 2;
    attribute mti_svvh_generic_type of MAX_LEN : constant is 2;
    attribute mti_svvh_generic_type of IDX_W : constant is 2;
end softmax_max;
