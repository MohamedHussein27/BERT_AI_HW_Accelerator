module write_logic_gen #(
    parameter ADDR_WIDTH          = 16,
    parameter MAX_DEPTH           = 36864 // max depth of the BRAM
)(
    // System Signals
    input  wire                      clk,
    input  wire                      rst_n,

    // Control Signals
    input  wire                      start_write,         // Pulse to begin writing the next tile
    input  wire                      reset_addr_counter,  // Pulse to reset the internal address pointer

    // BRAM Interface
    output wire  [ADDR_WIDTH-1:0]    bram_addr,           // Address sent to the BRAM
    output reg                       bram_we,             // Write enable for BRAM (acts as both en + we)

    // Status Signal
    output reg                       write_done,           // Pulse high for one cycle when done writing in a buffer
    output reg                       busy                  // signal to state we are writing at the moment
);

    // =====================
    // State Machine Definition
    // =====================
    localparam [1:0] IDLE     = 2'b00;
    localparam [1:0] WRITING  = 2'b01;
    localparam [1:0] DONE     = 2'b10;

    reg [1:0] current_state, next_state;

    // =====================
    // Internal Registers
    // =====================
    localparam COUNTER_WIDTH = $clog2(MAX_DEPTH);
    reg [COUNTER_WIDTH-1:0] write_offset;

    // =====================
    // Sequential Logic
    // =====================
    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            current_state <= IDLE;
            write_offset  <= 0;
        end else begin
            current_state <= next_state;

            if (reset_addr_counter)
                write_offset <= 0;

            if (current_state == WRITING)
                write_offset <= write_offset + 1;
        end
    end

    // =====================
    // Combinational Logic
    // =====================
    always @(*) begin
        next_state = current_state;
        bram_we    = 1'b0;
        write_done = 1'b0;
        busy       = 1'b0;

        case (current_state)
            IDLE: begin
                if (start_write)
                    next_state = WRITING;
            end

            WRITING: begin
                bram_we = 1'b1;
                busy    = 1'b1;
                if (bram_addr != MAX_DEPTH-1 && start_write) begin
                    next_state = WRITING;
                end
                else begin
                    next_state = DONE;
                end
            end

            DONE: begin
                write_done = 1'b1;
                next_state = IDLE;
            end

            default: next_state = IDLE;
        endcase
    end

    // output address 
    assign bram_addr  = write_offset;

endmodule
