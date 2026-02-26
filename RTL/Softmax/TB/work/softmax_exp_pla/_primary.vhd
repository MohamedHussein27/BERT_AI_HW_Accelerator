library verilog;
use verilog.vl_types.all;
entity softmax_exp_pla is
    generic(
        D_W             : integer := 32;
        FRAC_I          : integer := 26;
        O_W             : integer := 16;
        FRAC_O          : integer := 15;
        NSEG            : integer := 32;
        COEFF_W         : integer := 16;
        COEFF_F         : integer := 15;
        H_SHIFT         : integer := 25
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
    attribute mti_svvh_generic_type of D_W : constant is 2;
    attribute mti_svvh_generic_type of FRAC_I : constant is 2;
    attribute mti_svvh_generic_type of O_W : constant is 2;
    attribute mti_svvh_generic_type of FRAC_O : constant is 2;
    attribute mti_svvh_generic_type of NSEG : constant is 2;
    attribute mti_svvh_generic_type of COEFF_W : constant is 2;
    attribute mti_svvh_generic_type of COEFF_F : constant is 2;
    attribute mti_svvh_generic_type of H_SHIFT : constant is 2;
end softmax_exp_pla;
