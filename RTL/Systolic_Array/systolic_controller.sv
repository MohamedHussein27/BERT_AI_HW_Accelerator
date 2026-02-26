module systolic_controller #(
    parameter DATAWIDTH    = 8,
    parameter N_SIZE       = 32,
    parameter num_of_raws  = 512,
    parameter BUS_WIDTH    = 256,   // N_SIZE * DATAWIDTH
    parameter ADDR_WIDTH   = 10
) (
    input  logic clk,
    input  logic rst_n,
    input  logic valid_in,
    input  logic load_weight,

    // Control signals to the Systolic Array
    output logic sys_wt_en,
    output logic we,
    output logic [ADDR_WIDTH-1:0]       rd_addr,
    output logic [ADDR_WIDTH-1:0]       wr_addr,
    output logic [$clog2(N_SIZE)-1:0]   wt_row_sel,

    // Status signals
    output logic ready,
    output logic busy,
    output logic done
);

    localparam [1:0] IDLE = 2'b00;
    localparam [1:0] LOADING_WEIGHT = 2'b01;
    localparam [1:0] COMPUTE = 2'b10;

    localparam count_bits = $clog2(num_of_raws + 2*N_SIZE);
    localparam wt_count_bits = $clog2(N_SIZE);

    logic [1:0] cs, ns;
    logic [count_bits-1:0] cycle_cnt;
    logic [wt_count_bits-1:0] wt_load_cnt;

    // Internal address accumulators
    logic [ADDR_WIDTH-1:0] rd_addr_reg;
    logic [ADDR_WIDTH-1:0] wr_addr_reg;


    // State Register
    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) cs <= IDLE;
        else        cs <= ns;
    end

    // next state logic
    always_comb begin
        ns = cs;
        case (cs)
            IDLE: begin
                if      (load_weight) ns = LOADING_WEIGHT;
                else if (valid_in)    ns = COMPUTE;
            end

            LOADING_WEIGHT: begin
                if (!load_weight || (wt_load_cnt == N_SIZE - 1))
                    ns = IDLE;
            end

            COMPUTE: begin
                if (cycle_cnt == num_of_raws + 2*N_SIZE - 2)
                    ns = IDLE;
            end

            default: ns = IDLE;
        endcase
    end

    // output logic
    always_comb begin
        case (cs)
            IDLE: begin
                sys_wt_en = 1'b0;
                rd_addr   = (valid_in)? 1 : '0;
                wr_addr   = '0;
                we        = 1'b0;
                done      = 1'b0;
                busy      = 1'b0;
                ready     = 1'b1;
            end

            LOADING_WEIGHT: begin
                sys_wt_en = 1'b1;
                rd_addr   = '0;
                wr_addr   = '0;
                we        = 1'b0;
                done      = 1'b0;
                busy      = 1'b1;
                ready     = 1'b0;
            end

            COMPUTE: begin
                sys_wt_en = 1'b0;
                rd_addr   = rd_addr_reg + 1;
                wr_addr   = wr_addr_reg;
                we   = (cycle_cnt >= N_SIZE-1) && (cycle_cnt < num_of_raws + 2*N_SIZE - 2);
                done = (cycle_cnt == num_of_raws + 2*N_SIZE - 2);
                busy  = 1'b1;
                ready = 1'b0;
            end
        endcase
    end

    // Address Registers
    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            rd_addr_reg <= '0;
            wr_addr_reg <= '0;
        end
        else begin
            case (cs)
                IDLE: begin
                    rd_addr_reg <= (valid_in)? rd_addr_reg + 1'b1 : '0;
                    wr_addr_reg <= '0;
                end

                COMPUTE: begin
                    if (rd_addr_reg < num_of_raws + N_SIZE - 2)
                        rd_addr_reg <= rd_addr_reg + 1'b1;
                    if (we)
                        wr_addr_reg <= wr_addr_reg + 1'b1;
                end
            endcase
        end
    end

    // Cycle counter
    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n)
            cycle_cnt <= '0;
        else if (cs == COMPUTE && valid_in)
            cycle_cnt <= cycle_cnt + 1'b1;
        else if (cs == IDLE)
            cycle_cnt <= '0;
    end

    // Weight-loading counter
    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n)
            wt_load_cnt <= '0;
        else if (cs == LOADING_WEIGHT)
            wt_load_cnt <= wt_load_cnt + 1'b1;
        else
            wt_load_cnt <= '0;
    end

    assign wt_row_sel = wt_load_cnt;
endmodule