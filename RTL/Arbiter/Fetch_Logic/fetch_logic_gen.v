module fetch_logic_gen #(
    parameter ADDR_WIDTH           = 16,
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
    input  wire [3:0]                   Buffer_Select,       // control signal to choose which buffer we are reading from, you will find it at the end of the code
    input  wire                         Tiles_Control,       // control signal to choose how many reads we will do, if weights so we will tile 32 times, if inputs we will tile 512 times
    input  wire                         Double_buffering,    // control signal to make us read from the double buffering addresses in weights and inputs 

    // BRAM Interface
    output reg  [ADDR_WIDTH-1:0]        bram_addr,           // Address sent to the BRAM
    output reg                          bram_en,             // Enable signal for the BRAM read port

    // Status Signal
    output reg                          fetch_done,           // Pulse high for one cycle when done
    output reg                          busy                  // busy signal to indicate we are still fetching
);

    // State Machine Definition
    localparam [1:0] IDLE     = 2'b00;
    localparam [1:0] FETCHING = 2'b01;
    localparam [1:0] DONE     = 2'b10;

    reg [1:0] current_state, next_state;
    
    reg [14:0] FETCH_START_OFFSET;
    wire [9:0] NUM_FETCHES_PER_TILE; // to choose how many fetchs we take in one tile
    
    integer i = 0, j = 0; // rows, columns

    // Internal Address Pointer
    // Pointer to track the next tile index to fetch.
    // The largest buffer might need 384 tiles, so 9 bits are needed.
    reg [9:0] addr_ptr;
    reg [8:0] transpose_addr_pointer; // pointer used only in K matrix as we need to seperate the pointers

    // Internal Counter for fetch offset within a tile
    reg [9:0] fetch_offset;

    // Sequential Logic
    always @(posedge clk or negedge rst_n)
        begin
            if (!rst_n) 
                begin
                    current_state          <= IDLE;
                    fetch_offset           <= 0;
                    addr_ptr               <= 0;
                    transpose_addr_pointer <= 0;
                end 
            else 
                begin
                    current_state <= next_state;
                    // equation for transposition, you can try iy on smaller matrices
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
                    // this condition is to reset the i (row) counter and j (column) counter in case the tile is not finished and the rows have reached their max  
                    else if (bram_addr == ((2 * FETCH_START_OFFSET) + addr_ptr) - ORIGINAL_COLUMNS && FETCH_START_OFFSET == ((ORIGINAL_COLUMNS*ORIGINAL_ROWS*NUM_BITS)/DATA_WIDTH))
                        begin
                            transpose_addr_pointer <= transpose_addr_pointer + 1;
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
                            transpose_addr_pointer <= 0;
                        end
                // Increment the pointer AFTER a fetch completes
                    else if (current_state == DONE && Buffer_Select != 3'b011) // as not to incerement pointers after a Kt fetching ( 3'b011 is Q but we know that Q comes after Kt)
                        begin
                            transpose_addr_pointer <= transpose_addr_pointer + 1;
                            addr_ptr               <= addr_ptr + 1;
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
            busy       = 1'b0;

            // The address is the current pointer value (tile index) multiplied by
            // the number of reads per tile, plus the intra-tile offset.
            if (FETCH_START_OFFSET != ((ORIGINAL_COLUMNS*ORIGINAL_ROWS*NUM_BITS)/DATA_WIDTH))
                begin
                    bram_addr  = (addr_ptr * NUM_FETCHES_PER_TILE) + fetch_offset + FETCH_START_OFFSET; // offset to account for the different address mapping in a single BRAM 
                end
            
            else
                begin
                    bram_addr = ORIGINAL_COLUMNS * j + i + transpose_addr_pointer + FETCH_START_OFFSET;
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
                    busy    = 1'b1;
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
        // always block to choose from which address will we start
        always @(*) begin
            case (Buffer_Select)

                // =====================================================
                // ================= ATTENTION BUFFERS =================
                // =====================================================

                4'b0000 : FETCH_START_OFFSET = 
                            (Double_buffering) ? 16'd32 : 16'd0;    // W

                4'b0001 : FETCH_START_OFFSET = 16'd64;              // b

                4'b0010 : FETCH_START_OFFSET = 
                            (Double_buffering) ? 16'd624 : 16'd112;  // I

                4'b0011 : FETCH_START_OFFSET = 16'd0;               // Q

                4'b0100 : FETCH_START_OFFSET = 
                    ((ORIGINAL_COLUMNS*ORIGINAL_ROWS*NUM_BITS)/DATA_WIDTH); // K

                4'b0101 : FETCH_START_OFFSET = 
                    2*((ORIGINAL_COLUMNS*ORIGINAL_ROWS*NUM_BITS)/DATA_WIDTH); // V

                4'b0110 : FETCH_START_OFFSET = 16'd8192;            // kTQ

                4'b0111 : FETCH_START_OFFSET = 16'd20480;           // SV

                4'b1000 : FETCH_START_OFFSET = 16'd28672;           // H


                // =====================================================
                // ==================== FFN BUFFERS ====================
                // =====================================================

                // FFN Weights (double buffering ONLY here)
                4'b1001 : FETCH_START_OFFSET =
                            (Double_buffering) ? 16'd32 : 16'd0;

                // FFN Bias
                4'b1010 : FETCH_START_OFFSET = 16'd64;

                // FFN Input
                4'b1011 : FETCH_START_OFFSET = 16'd160;

                // FFN Intermediate
                4'b1100 : FETCH_START_OFFSET = 16'd0;

                // Output Buffer (O_buffer)
                4'b1101 : FETCH_START_OFFSET = 16'd49152;

                default : FETCH_START_OFFSET = 16'd0;

            endcase
        end

        // if we are fetching weights, NUM_FETCHES_PER_TILE will be 32, if inputs NUM_FETCHES_PER_TILE will be 512
        assign NUM_FETCHES_PER_TILE = Tiles_Control ? 10'd32 : 10'd512;
endmodule

