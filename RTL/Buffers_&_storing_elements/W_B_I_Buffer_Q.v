`timescale 1ns / 1ps

module W_B_I_Buffer (
    // Port A: 256-bit Write Port (Modified from 32-bit)
    input  wire         clka,
    input  wire         ena,
    input  wire [0:0]   wea,
    input  wire [10:0]  addra,   // Changed from [13:0] to [10:0]
    input  wire [255:0] dina,    // Changed from [31:0] to [255:0]

    // Port B: 256-bit Read Port (Unchanged)
    input  wire         clkb,
    input  wire         enb,
    input  wire [10:0]  addrb,
    output reg  [255:0] doutb
);

    // =========================================================
    // Memory Array Definition
    // =========================================================
    // The data is now 256 bits wide. 
    // We size the array to 2048 (2^11) to perfectly match the 
    // new 11-bit address bus.
    reg [255:0] ram [0:2047];

    // =========================================================
    // PORT A: Write Logic (256-bit)
    // =========================================================
    always @(posedge clka) begin
        if (ena) begin
            if (wea) begin
                ram[addra] <= dina;
            end
        end
    end

    // =========================================================
    // PORT B: Read Logic (256-bit)
    // =========================================================
    always @(posedge clkb) begin
        if (enb) begin
            doutb <= ram[addrb];
        end
    end

endmodule