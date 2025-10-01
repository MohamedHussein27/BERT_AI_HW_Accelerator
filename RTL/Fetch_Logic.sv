// fetch_logic.sv
// Fetch logic: read a weight tile (W_ROWS x W_COLS) and an input tile (I_ROWS x I_COLS)
// into internal buffers and stream them to a systolic array using ready/valid handshake.

module fetch_logic #(
    parameter int DATA_W   = 16,
    parameter int W_ROWS   = 32,
    parameter int W_COLS   = 32,
    parameter int I_ROWS   = 512,
    parameter int I_COLS   = 32,
    parameter int ADDR_W   = 32  // width of memory addresses
)(
    input  logic                     clk,
    input  logic                     rst_n,

    // control
    input  logic                     start,      // start fetching a pair of tiles
    output logic                     busy,
    output logic                     done,       // asserted for one cycle when a pair completed

    // weight memory read interface (external memory / arbiter)
    output logic                     w_rd_en,
    output logic [ADDR_W-1:0]        w_rd_addr,
    input  logic [DATA_W-1:0]        w_rd_data,   // synchronous read data (valid next cycle)

    // input memory read interface (external memory / arbiter)
    output logic                     i_rd_en,
    output logic [ADDR_W-1:0]        i_rd_addr,
    input  logic [DATA_W-1:0]        i_rd_data,   // synchronous read data (valid next cycle)

    // streaming output to SA (serial element streaming)
    output logic                     out_valid,
    input  logic                     out_ready,
    output logic [DATA_W-1:0]        out_data,
    output logic                     out_last
);

    // Derived sizes
    localparam int W_NELEMS = W_ROWS * W_COLS;
    localparam int I_NELEMS = I_ROWS * I_COLS;
    localparam int TOTAL_NELEMS = W_NELEMS + I_NELEMS;

    // Simple address mapping: caller provides base addresses for weight & input
    // For generality, we'll have internal base regs set by the arbiter when start is asserted.
    // Here we use internally incrementing addresses starting from provided inputs (could be ports).
    // For this example, start triggers reading from addresses 0..N-1.
    // You should connect w_addr_base/i_addr_base via additional ports if needed.

    // Internal buffer memories (synthesizable arrays)
    // Note: large arrays infer block RAMs in most flows.
    logic [DATA_W-1:0] tileA_mem [0:W_NELEMS-1]; // weights 32x32
    logic [DATA_W-1:0] tileB_mem [0:I_NELEMS-1]; // inputs 512x32

    // FSM states
    typedef enum logic [1:0] {IDLE=2'd0, FETCH_W=2'd1, FETCH_I=2'd2, STREAM=2'd3} state_t;
    state_t state, next_state;

    // counters
    logic [$clog2(W_NELEMS+1)-1:0] w_cnt;
    logic [$clog2(I_NELEMS+1)-1:0] i_cnt;
    logic [$clog2(TOTAL_NELEMS+1)-1:0] s_cnt; // streaming counter

    // Read request registers (we issue rd_en for one cycle; data appears next cycle)
    logic w_rd_pending;
    logic i_rd_pending;
    logic [ADDR_W-1:0] w_addr_reg;
    logic [ADDR_W-1:0] i_addr_reg;

    // Output registers for next-cycle data capture
    logic [DATA_W-1:0] rd_data_latched;

    // Busy signal
    assign busy = (state != IDLE);

    // Default outputs
    assign w_rd_en = w_rd_pending;
    assign w_rd_addr = w_addr_reg;
    assign i_rd_en = i_rd_pending;
    assign i_rd_addr = i_addr_reg;

    // Next-state combinational
    always_comb begin
        next_state = state;
        case (state)
            IDLE: if (start) next_state = FETCH_W;
            FETCH_W: if (w_cnt == W_NELEMS && !w_rd_pending) next_state = FETCH_I;
            FETCH_I: if (i_cnt == I_NELEMS && !i_rd_pending) next_state = STREAM;
            STREAM: if (s_cnt == TOTAL_NELEMS && !out_valid) next_state = IDLE; // done after last word accepted
            default: next_state = IDLE;
        endcase
    end

    // FSM sequential and counters
    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            state <= IDLE;
            w_cnt <= '0;
            i_cnt <= '0;
            s_cnt <= '0;
            w_rd_pending <= 1'b0;
            i_rd_pending <= 1'b0;
            w_addr_reg <= '0;
            i_addr_reg <= '0;
            out_valid <= 1'b0;
            done <= 1'b0;
        end else begin
            state <= next_state;
            done <= 1'b0;

            case (state)
                IDLE: begin
                    w_cnt <= 0;
                    i_cnt <= 0;
                    s_cnt <= 0;
                    w_rd_pending <= 1'b0;
                    i_rd_pending <= 1'b0;
                    out_valid <= 1'b0;
                    // base addresses could be set here if they were ports
                    w_addr_reg <= '0;
                    i_addr_reg <= '0;
                end

                FETCH_W: begin
                    // If we still have weight elements to fetch and no outstanding request, issue one
                    if (w_cnt < W_NELEMS && !w_rd_pending) begin
                        w_rd_pending <= 1'b1;
                        w_addr_reg <= w_cnt; // simply linear addresses 0..W_NELEMS-1
                    end
                    // Data returns next cycle -> capture
                    if (w_rd_pending) begin
                        // Capture returned data into tileA_mem[w_cnt]
                        tileA_mem[w_cnt] <= w_rd_data;
                        w_rd_pending <= 1'b0;
                        w_cnt <= w_cnt + 1;
                    end
                end

                FETCH_I: begin
                    if (i_cnt < I_NELEMS && !i_rd_pending) begin
                        i_rd_pending <= 1'b1;
                        i_addr_reg <= i_cnt; // addresses 0..I_NELEMS-1 (or offset by base)
                    end
                    if (i_rd_pending) begin
                        tileB_mem[i_cnt] <= i_rd_data;
                        i_rd_pending <= 1'b0;
                        i_cnt <= i_cnt + 1;
                    end
                end

                STREAM: begin
                    // prepare output valid if SA ready to accept (or just assert valid and wait for ready)
                    if (!out_valid) begin
                        out_valid <= 1'b1;
                    end

                    if (out_valid && out_ready) begin
                        s_cnt <= s_cnt + 1;
                        // if we've streamed the last element
                        if (s_cnt + 1 == TOTAL_NELEMS) begin
                            out_valid <= 1'b0; // after last accepted, lower valid
                            done <= 1'b1;      // single-cycle pulse
                        end
                    end
                end

            endcase
        end
    end

    // Read data mux for streaming:
    // stream order: first W_NELEMS elements from tileA_mem then I_NELEMS from tileB_mem
    // out_data is valid during out_valid and when SA asserts out_ready it consumes element.
    // We output the current element indexed by s_cnt (but must careful with read-before-write; s_cnt increments after handshake).
    logic [31:0] s_index;
    always_comb begin
        s_index = s_cnt;
        if (s_index < W_NELEMS) begin
            out_data = tileA_mem[s_index];
        end else begin
            out_data = tileB_mem[s_index - W_NELEMS];
        end
    end

    // out_last pulses when the next accepted element is the last
    assign out_last = (out_valid && (s_cnt + 1 == TOTAL_NELEMS) && out_ready);

endmodule
