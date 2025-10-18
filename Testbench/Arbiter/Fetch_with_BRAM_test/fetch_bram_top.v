module fetch_bram_top (
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
    input  wire [1:0]   buffer_select, 

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
    output wire [255:0] doutb,      // data read from BRAM (Port B)
    output wire [10:0]  addrb       // address used for read (for debug)
);

    // =====================
    // Internal signals
    // =====================
    wire bram_en_b;

    // =====================
    // Instantiate the Fetch Logic
    // =====================
    fetch_logic_gen #(
        .NUM_FETCHES_PER_TILE(32),
        .ADDR_WIDTH(11)
    ) u_fetch_logic (
        .clk(clk),
        .rst_n(rst_n),

        .start_fetch(start_fetch),
        .reset_addr_counter(reset_addr_counter),

        //.buffer_select(buffer_select),

        .bram_addr(addrb),
        .bram_en(bram_en_b),

        .fetch_done(fetch_done)
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
