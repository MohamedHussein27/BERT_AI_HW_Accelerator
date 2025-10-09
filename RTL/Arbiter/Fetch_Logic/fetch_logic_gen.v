module fetch_logic_gen #(
    parameter NUM_FETCHES_PER_TILE = 2,
    parameter ADDR_WIDTH           = 11
) (
    // System Signals
    input  wire                         clk,
    input  wire                         rst_n,

    // Control Signals
    input  wire                         start_fetch,         // Pulse to begin fetching the next tile
    input  wire                         reset_addr_counter,  // Pulse to reset the internal address pointer

    // BRAM Interface
    output reg  [ADDR_WIDTH-1:0]        bram_addr,           // Address sent to the BRAM
    output reg                          bram_en,             // Enable signal for the BRAM read port

    // Status Signal
    output reg                          fetch_done           // Pulse high for one cycle when done
);

    // State Machine Definition
    localparam [1:0] IDLE     = 2'b00;
    localparam [1:0] FETCHING = 2'b01;
    localparam [1:0] DONE     = 2'b10;

    reg [1:0] current_state, next_state;

    // Internal Address Pointer
    // Pointer to track the next tile index to fetch.
    // The largest buffer might need 384 tiles, so 9 bits are needed.
    reg [8:0] addr_ptr;

    // Internal Counter for fetch offset within a tile
    localparam COUNTER_WIDTH = $clog2(NUM_FETCHES_PER_TILE);
    reg [COUNTER_WIDTH-1:0] fetch_offset;

    // Sequential Logic
    always @(posedge clk or negedge rst_n) 
        begin
            if (!rst_n) 
                begin
                    current_state <= IDLE;
                    fetch_offset <= 0;
                    addr_ptr     <= 0;
                end 
            else 
                begin
                    current_state <= next_state;

                // Reset address pointer if requested
                if (reset_addr_counter) 
                    begin
                        addr_ptr <= 0;
                    end
                // Increment the pointer AFTER a fetch completes
                else if (state == DONE) 
                    begin
                        addr_ptr <= addr_ptr + 1;
                    end

                // Counter logic for fetches within a tile
                if (next_state == IDLE) 
                    begin
                        fetch_offset <= 0;
                    end 
                else if (state == FETCHING) 
                    begin
                        fetch_offset <= fetch_offset + 1;
                    end
            end
        end

    // Combinational Logic
    always @(*) 
        begin
            next_state = state;
            bram_en    = 1'b0;
            fetch_done = 1'b0;

            // The address is the current pointer value (tile index) multiplied by
            // the number of reads per tile, plus the intra-tile offset.
            bram_addr  = (addr_ptr * NUM_FETCHES_PER_TILE) + fetch_offset;

            case (state)
                IDLE: begin
                    if (start_fetch) 
                        begin
                            next_state = FETCHING;
                        end
                      end
                FETCHING: begin
                    bram_en = 1'b1;
                    if (fetch_offset == NUM_FETCHES_PER_TILE - 1) 
                        begin
                            next_state = DONE;
                        end
                          end
                DONE: begin
                    fetch_done = 1'b1;
                    next_state = IDLE;
                end
                default: begin
                    next_state = IDLE;
                end
            endcase
        end

endmodule