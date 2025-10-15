module write_bram_top (
    // =====================
    // System signals
    // =====================
    input  wire         clk,
    input  wire         rst_n,

    // =====================
    // Control signals for the write logic
    // =====================
    input  wire         start_write,         // pulse to start writing a new tile
    input  wire         reset_addr_counter,  // resets internal tile counter

    // =====================
    // Data input from Systolic Array
    // =====================
    input  wire [255:0] sa_out_data,         // output data from systolic array

    // =====================
    // Optional read port (Port B)
    // =====================
    input  wire         read_en,             // enable read from port B
    input  wire [15:0]  read_addr,           // address to read (for test or debug)
    output wire [255:0] doutb,               // data read from BRAM

    // =====================
    // Status outputs
    // =====================
    output wire         write_done,          // pulse high when tile write complete
    output wire [15:0]  current_addr         // optional: for monitoring/debug
);

    // Internal signals
    wire [15:0] bram_addr;
    wire        bram_we;

    // =====================
    // Write logic instantiation
    // =====================
    write_logic_gen #(
        .NUM_WRITES_PER_TILE(16),
        .ADDR_WIDTH(16),
        .ADDR_STRIDE(23)
    ) u_write_logic (
        .clk(clk),
        .rst_n(rst_n),

        .start_write(start_write),
        .reset_addr_counter(reset_addr_counter),

        .bram_addr(bram_addr),
        .bram_we(bram_we),

        .write_done(write_done)
    );

    // =====================
    // BRAM instantiation (Q_K_V_buffer)
    // =====================
    Q_K_V_buffer q_k_v_bram (
        // Port A (write from SA)
        .clka(clk),
        .ena(bram_we),           // use write enable as enable (active when writing)
        .wea(bram_we),           // single-bit write enable
        .addra(bram_addr),       // address from write logic
        .dina(sa_out_data),      // data from systolic array (256 bits)

        // Port B (optional read/debug)
        .clkb(clk),
        .enb(read_en),
        .addrb(read_addr),
        .doutb(doutb)
    );

    // =====================
    // Debug/monitor output
    // =====================
    assign current_addr = bram_addr;

endmodule
