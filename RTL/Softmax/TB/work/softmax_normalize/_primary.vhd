library verilog;
use verilog.vl_types.all;
entity softmax_normalize is
    generic(
        EXP_WIDTH       : integer := 16;
        FRAC_E          : integer := 15;
        RCP_WIDTH       : integer := 32;
        FRAC_R          : integer := 24;
        OUT_WIDTH       : integer := 16;
        FRAC_OUT        : integer := 15;
        MAX_LEN         : integer := 512;
        IDX_W           : integer := 10
    );
    port(
        clk             : in     vl_logic;
        rst_n           : in     vl_logic;
        start           : in     vl_logic;
        vec_len_cfg     : in     vl_logic_vector;
        reciprocal      : in     vl_logic_vector;
        exp_rd_addr     : out    vl_logic_vector;
        exp_rd_data     : in     vl_logic_vector;
        out_valid       : out    vl_logic;
        out_data        : out    vl_logic_vector;
        out_last        : out    vl_logic
    );
    attribute mti_svvh_generic_type : integer;
    attribute mti_svvh_generic_type of EXP_WIDTH : constant is 2;
    attribute mti_svvh_generic_type of FRAC_E : constant is 2;
    attribute mti_svvh_generic_type of RCP_WIDTH : constant is 2;
    attribute mti_svvh_generic_type of FRAC_R : constant is 2;
    attribute mti_svvh_generic_type of OUT_WIDTH : constant is 2;
    attribute mti_svvh_generic_type of FRAC_OUT : constant is 2;
    attribute mti_svvh_generic_type of MAX_LEN : constant is 2;
    attribute mti_svvh_generic_type of IDX_W : constant is 2;
end softmax_normalize;
