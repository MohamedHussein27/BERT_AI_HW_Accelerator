`timescale 1ns / 1ps

module Q_Kt_buffer (
    // Port A: 256-bit Write Port
    input  wire         clka,
    input  wire         ena,
    input  wire [0:0]   wea,
    input  wire [12:0]  addra,
    input  wire [255:0] dina,

    // Port B: 256-bit Read Port
    input  wire         clkb,
    input  wire         enb,
    input  wire [12:0]  addrb,
    output reg  [255:0] doutb
);

    // =========================================================
    // Memory Array Definition
    // =========================================================
    // The data is 256 bits wide. 
    // We size the array to 8192 (2^13) to perfectly match the 
    // 13-bit address bus.
    reg [255:0] ram [0:8191];

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