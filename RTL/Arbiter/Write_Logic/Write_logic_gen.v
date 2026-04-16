module write_logic_gen #(
    parameter ADDR_WIDTH          = 16,
    parameter MAX_DEPTH           = 36864, // max depth of the BRAM
    parameter TILE_DEPTH          = 512,   // elements per tile
    parameter NUM_BUFFERS         = 12     // Total number of buffers in the system
)(
    // System Signals
    input  wire                      clk,
    input  wire                      rst_n,

    // Control Signals
    input  wire                      start_write,         // Pulse to begin writing the next tile
    input  wire                      reset_addr_counter,  // Pulse to reset the internal address pointer
    input  wire [3:0]                Buffer_Select,       // Chooses which buffer region to write to
    input  wire                      Double_buffering,    // For ping-pong buffer writing

    // sipo interface (softmax)
    input  wire                      sipo_valid_out,      // to just inceremnt the address when this signal is high
    input  wire                      sipo_mode,           // to make the write logic obeys sipo rules

    // BRAM Interface
    output wire [ADDR_WIDTH-1:0]     bram_addr,           // Address sent to the BRAM
    output reg  [NUM_BUFFERS-1:0]    bram_we,             // One-hot write enable for BRAMs (acts as both en + we)

    // Status Signal
    output reg                       write_all_done,      // Pulse high when the ENTIRE buffer (MAX_DEPTH) is written
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

    // Internal signal to indicate if the FSM wants to write right now
    reg general_we; 

    // =====================
    // Internal Registers
    // =====================
    localparam COUNTER_WIDTH      = $clog2(MAX_DEPTH);
    
    reg [COUNTER_WIDTH-1:0]      write_offset; // Tracks total writes for the buffer
    reg [7:0]                    tile_offset;  // Tracks writes for the current tile
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
                if (sipo_mode) begin
                    if (sipo_valid_out)
                        write_offset <= write_offset + 1; // making the write logic counts only when its valid data from sipo
                end
                else begin
                    write_offset <= write_offset + 1;
                end
                
                // incerement when hit the max if the tile
                if (write_offset == TILE_DEPTH - 1)
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
        general_we      = 1'b0;
        write_all_done  = 1'b0;
        write_tile_done = 1'b0;
        busy            = 1'b0;

        case (current_state)
            IDLE: begin
                if (start_write)
                    next_state = WRITING;
            end

            WRITING: begin
                general_we = 1'b1;
                busy       = 1'b1;
                
                if (sipo_mode && (tile_offset == 16 - 1)) begin // 16 = 512 / 32 which we will know that this is the last row to be written the SM buffer
                    next_state = DONE_ALL;
                end

                // Check if we just finished a standard tile
                else if (tile_offset == TILE_DEPTH - 1) begin
                    next_state = DONE;
                end
                
                // Check if we just finished the very last element of the entire buffer
                else if ((write_offset == MAX_DEPTH - 1) && !sipo_mode) begin
                    next_state = DONE_ALL;
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
                write_all_done = 1'b1;       // Signal Master Ctrl that the WHOLE BUFFER is done
                next_state = IDLE;
            end

            default: next_state = IDLE;
        endcase
    end

    // =====================
    // Buffer Select Address & Enable Mapping
    // =====================
    always @(*) begin
        // Default to all disables and address 0
        bram_we            = {NUM_BUFFERS{1'b0}}; 
        WRITE_START_OFFSET = 16'd0;

        case (Buffer_Select)
            4'b0000 : begin 
                WRITE_START_OFFSET = 16'd0;          // Q
                bram_we[0] = general_we;
            end
            4'b0001 : begin 
                WRITE_START_OFFSET = 16'd12288;      // K
                bram_we[1] = general_we;
            end
            4'b0010 : begin 
                WRITE_START_OFFSET = 16'd24576;      // V
                bram_we[2] = general_we;
            end
            4'b0011 : begin 
                WRITE_START_OFFSET = 16'd0;          // kTQ
                bram_we[3] = general_we;
            end
            4'b0100 : begin 
                WRITE_START_OFFSET = 16'd0;          // SM 
                bram_we[4] = general_we;
            end
            4'b0101 : begin 
                WRITE_START_OFFSET = 16'd0;          // SV
                bram_we[5] = general_we;
            end
            4'b0110 : begin 
                WRITE_START_OFFSET = 16'd12288;      // H
                bram_we[6] = general_we;
            end
            4'b0111 : begin 
                WRITE_START_OFFSET = 16'd0;          // LN buffer
                bram_we[7] = general_we;
            end

            // =====================================================
            // ==================== FFN BUFFERS ====================
            // =====================================================

            4'b1000 : begin 
                WRITE_START_OFFSET = (Double_buffering) ? 16'd12448 : 16'd0; // FFN Input
                bram_we[8] = general_we;
            end
            4'b1001 : begin 
                WRITE_START_OFFSET = 16'd0;          // FFN Intermediate
                bram_we[9] = general_we;
            end
            4'b1010 : begin 
                WRITE_START_OFFSET = (Double_buffering) ? 16'd61440 : 16'd0; // Output Buffer (O_buffer)
                bram_we[10] = general_we;
            end
            4'b1011 : begin 
                WRITE_START_OFFSET = 16'd0;          // FFN LN buffer
                bram_we[11] = general_we;
            end

            default : begin 
                WRITE_START_OFFSET = 16'd0;
                bram_we = {NUM_BUFFERS{1'b0}};
            end
        endcase
    end

    // =====================
    // Final Address Output
    // =====================
    // The physical address is the Base Offset + the Number of words written so far
    assign bram_addr = WRITE_START_OFFSET + write_offset;

endmodule