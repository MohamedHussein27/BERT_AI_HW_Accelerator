module fetch_logic #(
    parameter NUM_FETCHES_PER_TILE = 2,
    parameter ADDR_WIDTH           = 11
) (
    // System Signals
    input  wire                         clk,
    input  wire                         rst_n,

    // Control Signals
    input  wire                         start_fetch,         // Pulse to begin fetching the next tile
    input  wire                         reset_addr_counters, // Pulse to reset all internal address pointers
    input  wire [1:0]                   buffer_select,       // 00: Weights, 01: K-Matrix, 10: V-Matrix

    // BRAM Interface
    output reg  [ADDR_WIDTH-1:0]        bram_addr,           // Address sent to the BRAM
    output reg                          bram_en,             // Enable signal for the BRAM read port

    // Status Signal
    output reg                          fetch_done           // Pulse high for one cycle when done
);

    // Memory Map Definition
    localparam WEIGHT_BUFFER_BASE_ADDR = 11'd0;
    localparam K_MATRIX_BUFFER_BASE_ADDR = 11'd4;
    localparam V_MATRIX_BUFFER_BASE_ADDR = 11'd772;

    // State Machine Definition
    localparam [1:0] IDLE     = 2'b00;
    localparam [1:0] FETCHING = 2'b01;
    localparam [1:0] DONE     = 2'b10;

    reg [1:0] current_state, next_state;
    

    // Internal Address Pointers for each logical buffer
    // Pointers to track the next tile index for each buffer.
    reg [8:0] weight_addr_ptr;
    reg [8:0] k_matrix_addr_ptr;
    reg [8:0] v_matrix_addr_ptr;


    // Internal Counter for fetch offset within a tile
    localparam COUNTER_WIDTH = $clog2(NUM_FETCHES_PER_TILE);
    reg [COUNTER_WIDTH-1:0] fetch_offset;
    
    reg [ADDR_WIDTH-1:0] current_tile_base_addr;


    // Sequential Always Logic
    always @(posedge clk or negedge rst_n) 
        begin
            if (!rst_n) 
                begin
                    current_state        <= IDLE;
                    fetch_offset <= 0;
                    current_tile_base_addr <= 0;
                    weight_addr_ptr <= 0;
                    k_matrix_addr_ptr <= 0;
                    v_matrix_addr_ptr <= 0;
                end 
            else 
                begin
                    current_state <= next_state;
            
            // Reset all address pointers if requested
            if (reset_addr_counters) 
                begin
                    weight_addr_ptr <= 0;
                    k_matrix_addr_ptr <= 0;
                    v_matrix_addr_ptr <= 0;
                end
            // Increment the correct pointer AFTER a fetch completes
            else if (state == DONE) 
                begin
                    case (buffer_select)
                        2'b00: weight_addr_ptr <= weight_addr_ptr + 1;
                        2'b01: k_matrix_addr_ptr <= k_matrix_addr_ptr + 1;
                        2'b10: v_matrix_addr_ptr <= v_matrix_addr_ptr + 1;
                    endcase
                end
            
            // Latch the calculated base address when a fetch starts
            if (current_state == IDLE && next_state == FETCHING) 
                begin
                    case (buffer_select)
                        2'b00:  current_tile_base_addr <= WEIGHT_BUFFER_BASE_ADDR + (weight_addr_ptr * NUM_FETCHES_PER_TILE);
                        2'b01:  current_tile_base_addr <= K_MATRIX_BUFFER_BASE_ADDR + (k_matrix_addr_ptr * NUM_FETCHES_PER_TILE);
                        2'b10:  current_tile_base_addr <= V_MATRIX_BUFFER_BASE_ADDR + (v_matrix_addr_ptr * NUM_FETCHES_PER_TILE);
                        default: current_tile_base_addr <= 0;
                    endcase
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

    // Combinational Always Logic (States & Output Logic) 
    always @(*) 
        begin
            next_state = current_state;
            bram_en    = 1'b0;
            fetch_done = 1'b0;
            bram_addr  = current_tile_base_addr + fetch_offset;

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

