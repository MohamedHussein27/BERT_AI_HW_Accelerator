library verilog;
use verilog.vl_types.all;
library work;
entity softmax_control_fsm is
    generic(
        IDX_W           : integer := 10
    );
    port(
        clk             : in     vl_logic;
        rst_n           : in     vl_logic;
        start           : in     vl_logic;
        vec_len_cfg     : in     vl_logic_vector;
        in_valid        : in     vl_logic;
        in_ready        : out    vl_logic;
        max_done        : in     vl_logic;
        sub_exp_last    : in     vl_logic;
        acc_sum_valid   : in     vl_logic;
        recip_done      : in     vl_logic;
        norm_last       : in     vl_logic;
        max_start       : out    vl_logic;
        sub_start       : out    vl_logic;
        acc_start       : out    vl_logic;
        recip_start     : out    vl_logic;
        norm_start      : out    vl_logic;
        busy            : out    vl_logic;
        done            : out    vl_logic;
        current_state   : out    work.softmax_pkg.sm_state_t
    );
    attribute mti_svvh_generic_type : integer;
    attribute mti_svvh_generic_type of IDX_W : constant is 2;
end softmax_control_fsm;
