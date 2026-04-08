`timescale 1ns / 1ps

module W_B_I_Buffer (
    // Port A: 32-bit Write Port
    input  wire         clka,
    input  wire         ena,
    input  wire [0:0]   wea,
    input  wire [13:0]  addra,
    input  wire [31:0]  dina,

    // Port B: 256-bit Read Port
    input  wire         clkb,
    input  wire         enb,
    input  wire [10:0]  addrb,
    output reg  [255:0] doutb
);

    // =========================================================
    // Memory Array Definition
    // =========================================================
    // We define the memory based on the narrower port (32-bits).
    // While your actual data depth is 9088, we size the array to 
    // 16384 (2^14) to match the 14-bit address bus and prevent 
    // "out of bounds" warnings in QuestaSim.
    reg [31:0] ram [0:16383];

    // =========================================================
    // PORT A: Write Logic (32-bit)
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
    // The 256-bit read port fetches 8 contiguous 32-bit words.
    // In Xilinx BRAM mapping, the LSBs (31:0) come from the base 
    // address, and the MSBs come from the highest address.
    always @(posedge clkb) begin
        if (enb) begin
            doutb <= {
                ram[{addrb, 3'd7}], // Bits [255:224]
                ram[{addrb, 3'd6}], // Bits [223:192]
                ram[{addrb, 3'd5}], // Bits [191:160]
                ram[{addrb, 3'd4}], // Bits [159:128]
                ram[{addrb, 3'd3}], // Bits [127:96]
                ram[{addrb, 3'd2}], // Bits [95:64]
                ram[{addrb, 3'd1}], // Bits [63:32]
                ram[{addrb, 3'd0}]  // Bits [31:0]
            };
        end
    end

endmodule