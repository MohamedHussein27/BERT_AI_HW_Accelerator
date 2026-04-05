module write_logic_gen #(
    parameter ADDR_WIDTH          = 16,
    parameter MAX_DEPTH           = 36864, // max depth of the BRAM
    parameter TILE_DEPTH          = 512    // elements per tile
)(
    // System Signals
    input  wire                      clk,
    input  wire                      rst_n,

    // Control Signals
    input  wire                      start_write,         // Pulse to begin writing the next tile
    input  wire                      reset_addr_counter,  // Pulse to reset the internal address pointer
    input  wire [3:0]                Buffer_Select,       // Chooses which buffer region to write to
    input  wire                      Double_buffering,    // For ping-pong buffer writing

    // BRAM Interface
    output wire [ADDR_WIDTH-1:0]     bram_addr,           // Address sent to the BRAM
    output reg                       bram_we,             // Write enable for BRAM (acts as both en + we)

    // Status Signal
    output reg                       write_done,          // Pulse high when the ENTIRE buffer (MAX_DEPTH) is written
    output reg                       write_tile_done,     // Pulse high when a SINGLE TILE is written
    output reg                       busy                 // Signal to state we are writing at the moment
);

    // =====================
    // State Machine Definition
    // =====================
    localparam [1:0] IDLE     = 2'b00;
    localparam [1:0] WRITING  = 2'b01;
    localparam [1:0] DONE     = 2'b10; // Tile finished
    localparam [1:0] DONE_ALL = 2'b11; // Entire buffer finished

    reg [1:0] current_state, next_state;

    // =====================
    // Internal Registers
    // =====================
    localparam COUNTER_WIDTH      = $clog2(MAX_DEPTH);
    localparam TILE_COUNTER_WIDTH = $clog2(TILE_DEPTH);
    
    reg [COUNTER_WIDTH-1:0]      write_offset; // Tracks total writes for the buffer
    reg [TILE_COUNTER_WIDTH-1:0] tile_offset;  // Tracks writes for the current tile
    reg [15:0]                   WRITE_START_OFFSET; // Base address from Buffer_Select

    // =====================
    // Sequential Logic (Counters & State)
    // =====================
    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            current_state <= IDLE;
            write_offset  <= 0;
            tile_offset   <= 0;
        end else begin
            current_state <= next_state;

            if (reset_addr_counter) begin
                write_offset <= 0;
                tile_offset  <= 0;
            end 
            else if (current_state == WRITING) begin
                write_offset <= write_offset + 1;
                
                // Wrap the tile counter when it hits the limit
                if (tile_offset == TILE_DEPTH - 1)
                    tile_offset <= 0;
                else
                    tile_offset <= tile_offset + 1;
            end
        end
    end

    // =====================
    // Combinational Logic (State Transitions & Outputs)
    // =====================
    always @(*) begin
        // Default values to prevent latches
        next_state      = current_state;
        bram_we         = 1'b0;
        write_done      = 1'b0;
        write_tile_done = 1'b0;
        busy            = 1'b0;

        case (current_state)
            IDLE: begin
                if (start_write)
                    next_state = WRITING;
            end

            WRITING: begin
                bram_we = 1'b1;
                busy    = 1'b1;
                
                // Check if we just finished the very last element of the entire buffer
                if (write_offset == MAX_DEPTH - 1) begin
                    next_state = DONE_ALL;
                end
                // Check if we just finished a standard tile
                else if (tile_offset == TILE_DEPTH - 1) begin
                    next_state = DONE;
                end
                // Otherwise, keep writing
                else begin
                    next_state = WRITING;
                end
            end

            DONE: begin
                write_tile_done = 1'b1; // Signal Master Ctrl that ONE tile is done
                next_state = IDLE;      // Wait for the next start_write pulse
            end

            DONE_ALL: begin
                write_done = 1'b1;      // Signal Master Ctrl that the WHOLE BUFFER is done
                next_state = IDLE;
            end

            default: next_state = IDLE;
        endcase
    end

    // =====================
    // Buffer Select Address Mapping
    // =====================
    always @(*) begin
        case (Buffer_Select)
            4'b0000 : WRITE_START_OFFSET = 16'd0;    // Q

            4'b0001 : WRITE_START_OFFSET = 16'12288; // K

            4'b0010 : WRITE_START_OFFSET = 16'24576; // V

            4'b0011 : WRITE_START_OFFSET = 0;                   // kTQ

            4'b0100 : WRITE_START_OFFSET = 0;                   // SV

            4'b0101 : WRITE_START_OFFSET = 16'd12288;            // H


            // =====================================================
            // ==================== FFN BUFFERS ====================
            // =====================================================

            // FFN Input
            4'b0110 : WRITE_START_OFFSET = 
                        (Double_buffering) ? 16'd12448 : 16'd0;

            // FFN Intermediate
            4'b0111 : WRITE_START_OFFSET = 16'd0;

            // Output Buffer (O_buffer)
            4'b1000 : WRITE_START_OFFSET = 
                        (Double_buffering) ? 16'd61440 : 16'd0;

                        
            default : WRITE_START_OFFSET = 16'd0;
        endcase
    end

    // =====================
    // Final Address Output
    // =====================
    // The physical address is the Base Offset + the Number of words written so far
    assign bram_addr = WRITE_START_OFFSET + write_offset;

endmodule