library verilog;
use verilog.vl_types.all;
entity bert_softmax is
    generic(
        VEC_LEN         : integer := 64;
        D_W             : integer := 32;
        O_W             : integer := 16;
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
        out_valid       : out    vl_logic;
        out_data        : out    vl_logic_vector;
        out_last        : out    vl_logic;
        busy            : out    vl_logic;
        done            : out    vl_logic
    );
    attribute mti_svvh_generic_type : integer;
    attribute mti_svvh_generic_type of VEC_LEN : constant is 2;
    attribute mti_svvh_generic_type of D_W : constant is 2;
    attribute mti_svvh_generic_type of O_W : constant is 2;
    attribute mti_svvh_generic_type of MAX_LEN : constant is 2;
    attribute mti_svvh_generic_type of IDX_W : constant is 2;
end bert_softmax;
