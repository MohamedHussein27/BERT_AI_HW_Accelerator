library verilog;
use verilog.vl_types.all;
entity softmax_subtract is
    generic(
        D_W             : integer := 32;
        IDX_W           : integer := 7
    );
    port(
        clk             : in     vl_logic;
        rst_n           : in     vl_logic;
        start           : in     vl_logic;
        vec_len_cfg     : in     vl_logic_vector;
        max_value       : in     vl_logic_vector;
        sram_rd_addr    : out    vl_logic_vector;
        sram_rd_data    : in     vl_logic_vector;
        out_valid       : out    vl_logic;
        out_data        : out    vl_logic_vector;
        out_last        : out    vl_logic
    );
    attribute mti_svvh_generic_type : integer;
    attribute mti_svvh_generic_type of D_W : constant is 2;
    attribute mti_svvh_generic_type of IDX_W : constant is 2;
end softmax_subtract;
