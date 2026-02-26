module fetch_bram_FFN_W_B_I_top #(
    parameter ADDR_WIDTH        = 16,
    parameter ORIGINAL_COLUMNS  = 768,
    parameter ORIGINAL_ROWS     = 512,
    parameter NUM_BITS          = 8,
    parameter DATA_WIDTH        = 256
)(
    // =====================
    // System signals
    // =====================
    input  wire                     clk,
    input  wire                     rst_n,

    // =====================
    // Control signals
    // =====================
    input  wire                     start_fetch,
    input  wire                     reset_addr_counter,
    input  wire [3:0]               Buffer_Select,
    input  wire                     Tiles_Control,
    input  wire                     Double_buffering,

    // =====================
    // Write-side (Port A)
    // =====================
    input  wire                     wea,
    input  wire                     ena,
    input  wire [13:0]              addra,
    input  wire [DATA_WIDTH-1:0]    dina,

    // =====================
    // Outputs
    // =====================
    output wire                     fetch_done,
    output wire                     busy,
    output wire [DATA_WIDTH-1:0]    doutb,
    output wire [ADDR_WIDTH-1:0]    addrb
);

    wire bram_en_b;

    // =====================
    // Fetch Logic
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
    // FFN BRAM
    // =====================
    FFN_W_B_I_buffer u_FFN_W_B_I_buffer (
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
