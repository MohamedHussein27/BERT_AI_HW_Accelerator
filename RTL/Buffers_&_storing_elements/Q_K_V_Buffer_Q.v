`timescale 1ns / 1ps

module Q_K_V_buffer (
    // Port A: 256-bit Write Port
    input  wire         clka,
    input  wire         ena,
    input  wire [0:0]   wea,
    input  wire [15:0]  addra,
    input  wire [255:0] dina,

    // Port B: 256-bit Read Port
    input  wire         clkb,
    input  wire         enb,
    input  wire [15:0]  addrb,
    output reg  [255:0] doutb
);

    // =========================================================
    // Memory Array Definition
    // =========================================================
    // The data is 256 bits wide. 
    // While your actual required depth is 36,864, we size the 
    // array to 65,536 (2^16) to match the 16-bit address bus. 
    // This prevents QuestaSim from throwing "index out of bounds" 
    // warnings if the address pointer temporarily wanders.
    reg [255:0] ram [0:65535];

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