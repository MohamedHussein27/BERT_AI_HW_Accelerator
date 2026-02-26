module fetch_bram_W_B_I_top #(
    parameter ADDR_WIDTH = 16,
    parameter ORIGINAL_COLUMNS     = 768,   // matrix columns before transpose
    parameter ORIGINAL_ROWS        = 512,   // matrix rows before transpose
    parameter NUM_BITS             = 8,     // quantized element
    parameter DATA_WIDTH           = 256    )  (
    // =====================
    // System signals
    // =====================
    input  wire         clk,
    input  wire         rst_n,

    // =====================
    // Control signals for fetch logic
    // =====================
    input  wire         start_fetch,
    input  wire         reset_addr_counter,
    input  wire [2:0]   Buffer_Select,
    input  wire         Tiles_Control,
    input  wire         Double_buffering,

    // =====================
    // Write-side (Port A) inputs to preload BRAM
    // =====================
    input  wire         wea,        // write enable (active high)
    input  wire         ena,        // enable for port A
    input  wire [13:0]  addra,      // write address
    input  wire [31:0]  dina,       // write data

    // =====================
    // Outputs for monitoring
    // =====================
    output wire         fetch_done, // pulse when tile done
    output wire         busy, // high wile fetchong
    output wire [DATA_WIDTH-1:0] doutb,      // data read from BRAM (Port B)
    output wire [ADDR_WIDTH-1:0]  addrb       // address used for read (for debug)
);

    // =====================
    // Internal signals
    // =====================
    wire bram_en_b;

    // =====================
    // Instantiate the Fetch Logic
    // =====================
    fetch_logic_gen #(
            .ADDR_WIDTH(ADDR_WIDTH),
            .ORIGINAL_COLUMNS(ORIGINAL_COLUMNS),
            .ORIGINAL_ROWS(ORIGINAL_ROWS),
            .NUM_BITS(NUM_BITS),
            .DATA_WIDTH(DATA_WIDTH)
        ) u_fetch_logic (
            .clk(clk),
            .rst_n(rst_n),
    
            .start_fetch(start_fetch),
            .reset_addr_counter(reset_addr_counter),
            .Buffer_Select(Buffer_Select),
            .Tiles_Control(Tiles_Control),
            .Double_buffering(Double_buffering),
    
            .bram_addr(addrb),
            .bram_en(bram_en_b),
    
            .fetch_done(fetch_done),
            .busy(busy)
        );


    // =====================
    // Instantiate the BRAM (Dual Port)
    // =====================
    W_B_I_Buffer w_b_i_bram (
        // Port A (write)
        .clka(clk),
        .ena(ena),
        .wea(wea),
        .addra(addra),
        .dina(dina),

        // Port B (read)
        .clkb(clk),
        .enb(bram_en_b),
        .addrb(addrb),
        .doutb(doutb)
    );

endmodule
