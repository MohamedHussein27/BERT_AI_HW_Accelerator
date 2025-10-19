module fetch_logic_gen #(
    parameter NUM_FETCHES_PER_TILE = 2,
    parameter ADDR_WIDTH           = 16,
    parameter FETCH_START_OFFSET   = 0,     // default offset is zero
    parameter ORIGINAL_COLUMNS     = 768,   // matrix columns before transpose
    parameter ORIGINAL_ROWS        = 512,   // matrix rows before transpose
    parameter NUM_BITS             = 8,     // quantized element
    parameter DATA_WIDTH           = 256
    
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
    
    //reg  [ADDR_WIDTH-1:0]        bram_addr_reg;
    
    integer i = 0, j = 0;

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
                    
                    if (FETCH_START_OFFSET == ((ORIGINAL_COLUMNS*ORIGINAL_ROWS*NUM_BITS)/DATA_WIDTH) && current_state == FETCHING && bram_addr < (2 * FETCH_START_OFFSET)- ORIGINAL_COLUMNS)
                        begin                        
                            if (j == ORIGINAL_ROWS - 1)
                                begin
                                    j <= 0;
                                    i <= i + 1;
                                end
                        
                            else
                                begin
                                    j <= j + 1;
                                end
                    end
                        
                    else if (bram_addr == (2 * FETCH_START_OFFSET)- ORIGINAL_COLUMNS && FETCH_START_OFFSET == ((ORIGINAL_COLUMNS*ORIGINAL_ROWS*NUM_BITS)/DATA_WIDTH))
                        begin
                            addr_ptr <= addr_ptr + 1;
                            i <= 0;
                            j <= 0;
                        end
                           
                    else
                        begin
                            i <= 0;
                            j <= 0;
                        end
                                    
                    if (reset_addr_counter) 
                        begin
                            addr_ptr <= 0;
                        end
                // Increment the pointer AFTER a fetch completes
                    else if (current_state == DONE) 
                        begin
                            addr_ptr <= addr_ptr + 1;
                        end

                // Counter logic for fetches within a tile
                    if (next_state == IDLE) 
                        begin
                            fetch_offset <= 0;
                        end 
                    else if (current_state == FETCHING) 
                        begin
                            fetch_offset <= fetch_offset + 1;
                        end
            end
        end

    // Combinational Logic
    always @(*) 
        begin
            next_state = current_state;
            bram_en    = 1'b0;
            fetch_done = 1'b0;

            // The address is the current pointer value (tile index) multiplied by
            // the number of reads per tile, plus the intra-tile offset.
            if (FETCH_START_OFFSET != ((ORIGINAL_COLUMNS*ORIGINAL_ROWS*NUM_BITS)/DATA_WIDTH))
                begin
                    bram_addr  = (addr_ptr * NUM_FETCHES_PER_TILE) + fetch_offset + FETCH_START_OFFSET; // offset to account for the different address mapping in a single BRAM 
                end
            
            else
                begin
                    bram_addr = ORIGINAL_COLUMNS * j + i + addr_ptr + FETCH_START_OFFSET;
                end
                         
            case (current_state)
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

