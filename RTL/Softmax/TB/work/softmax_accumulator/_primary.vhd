library verilog;
use verilog.vl_types.all;
entity softmax_accumulator is
    generic(
        I_W             : integer := 16;
        FRAC_I          : integer := 15;
        O_W             : integer := 32;
        FRAC_O          : integer := 24;
        MAX_LEN         : integer := 512;
        IDX_W           : integer := 10
    );
    port(
        clk             : in     vl_logic;
        rst_n           : in     vl_logic;
        start           : in     vl_logic;
        vec_len_cfg     : in     vl_logic_vector;
        in_valid        : in     vl_logic;
        in_data         : in     vl_logic_vector;
        sum_valid       : out    vl_logic;
        sum_out         : out    vl_logic_vector;
        rd_addr         : in     vl_logic_vector;
        rd_data         : out    vl_logic_vector
    );
    attribute mti_svvh_generic_type : integer;
    attribute mti_svvh_generic_type of I_W : constant is 2;
    attribute mti_svvh_generic_type of FRAC_I : constant is 2;
    attribute mti_svvh_generic_type of O_W : constant is 2;
    attribute mti_svvh_generic_type of FRAC_O : constant is 2;
    attribute mti_svvh_generic_type of MAX_LEN : constant is 2;
    attribute mti_svvh_generic_type of IDX_W : constant is 2;
end softmax_accumulator;
