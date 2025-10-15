module write_logic_gen #(
    parameter NUM_WRITES_PER_TILE = 16,
    parameter ADDR_WIDTH          = 16,
    parameter ADDR_STRIDE         = 24,   //  address step size (incerement by 24 places to accomodate the outputs ordering ftom the SA)
    parameter MAX_DEPTH           = 36864 // max depth of the BRAM
)(
    // System Signals
    input  wire                      clk,
    input  wire                      rst_n,

    // Control Signals
    input  wire                      start_write,         // Pulse to begin writing the next tile
    input  wire                      reset_addr_counter,  // Pulse to reset the internal address pointer

    // BRAM Interface
    output reg  [ADDR_WIDTH-1:0]     bram_addr,           // Address sent to the BRAM
    output reg                       bram_we,             // Write enable for BRAM (acts as both en + we)

    // Status Signal
    output reg                       write_done           // Pulse high for one cycle when done
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
    reg [8:0] addr_ptr;  // tile index
    localparam COUNTER_WIDTH = $clog2(NUM_WRITES_PER_TILE);
    reg [COUNTER_WIDTH-1:0] write_offset;
    reg flag;

    // =====================
    // Sequential Logic
    // =====================
    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            current_state <= IDLE;
            write_offset  <= 0;
            addr_ptr      <= 0;
        end else begin
            current_state <= next_state;

            // Reset address pointer if requested
            if (next_state == DONE)
                addr_ptr <= 0;
            /*else if (next_state == DONE)*/
            else if (flag)
                addr_ptr <= addr_ptr + 1;

            // Counter logic
            if (/*next_state == DONE*/ flag)
                write_offset <= 0;
            else if (current_state == WRITING)
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
        flag       = 1'b0;

        // Each tile starts at addr_ptr * (NUM_WRITES_PER_TILE * ADDR_STRIDE)
        // Each write within the tile increases by ADDR_STRIDE
        bram_addr  = addr_ptr + (write_offset * ADDR_STRIDE);

        case (current_state)
            IDLE: begin
                if (start_write)
                    next_state = WRITING;
            end

            WRITING: begin
                bram_we = 1'b1;
                if (write_offset == NUM_WRITES_PER_TILE - 1) begin
                    //next_state = DONE;
                    if (bram_addr != MAX_DEPTH-1) begin
                        next_state = WRITING;
                        flag = 1;
                    end
                    else 
                        next_state = DONE;
                end
                else 
                    next_state = WRITING;
            end

            DONE: begin
                write_done = 1'b1;
                next_state = IDLE;
                /*if (bram_addr == 384)
                    next_state = IDLE;
                else 
                    next_state = WRITING;*/
            end

            default: next_state = IDLE;
        endcase
    end

endmodule

