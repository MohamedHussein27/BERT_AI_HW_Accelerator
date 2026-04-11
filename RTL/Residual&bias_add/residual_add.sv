module vec_add_32 (
    input  wire [32*8-1:0] a,   // 32 x 8-bit signed inputs
    input  wire [32*8-1:0] b,
    output wire [32*8-1:0] sum  // 32 x 8-bit signed outputs
);
 
    genvar i;
    generate
        for (i = 0; i < 32; i = i + 1) begin : adder_array
            signed_adder_8bit u_add (
                .a   (a[i*8 +: 8]),
                .b   (b[i*8 +: 8]),
                .sum (sum[i*8 +: 8])  // only lower 8 bits connected
            );
        end
    endgenerate
 
endmodule
 
 

// Module : signed_adder_8bit
// 8-bit ripple carry adder, 9th bit (carry out) left unconnected

module signed_adder_8bit (
    input  wire [7:0] a,
    input  wire [7:0] b,
    output wire [8:0] sum   // sum[8] = carry out, ignored by top
);
 
    wire [8:0] c;
    wire [7:0] axb, anb, caxb;
 
    assign c[0] = 1'b0;
 
    assign axb[0]  = a[0] ^ b[0];  assign sum[0]  = axb[0] ^ c[0];
    assign anb[0]  = a[0] & b[0];  assign caxb[0] = c[0]   & axb[0];  assign c[1] = anb[0] | caxb[0];
 
    assign axb[1]  = a[1] ^ b[1];  assign sum[1]  = axb[1] ^ c[1];
    assign anb[1]  = a[1] & b[1];  assign caxb[1] = c[1]   & axb[1];  assign c[2] = anb[1] | caxb[1];
 
    assign axb[2]  = a[2] ^ b[2];  assign sum[2]  = axb[2] ^ c[2];
    assign anb[2]  = a[2] & b[2];  assign caxb[2] = c[2]   & axb[2];  assign c[3] = anb[2] | caxb[2];
 
    assign axb[3]  = a[3] ^ b[3];  assign sum[3]  = axb[3] ^ c[3];
    assign anb[3]  = a[3] & b[3];  assign caxb[3] = c[3]   & axb[3];  assign c[4] = anb[3] | caxb[3];
 
    assign axb[4]  = a[4] ^ b[4];  assign sum[4]  = axb[4] ^ c[4];
    assign anb[4]  = a[4] & b[4];  assign caxb[4] = c[4]   & axb[4];  assign c[5] = anb[4] | caxb[4];
 
    assign axb[5]  = a[5] ^ b[5];  assign sum[5]  = axb[5] ^ c[5];
    assign anb[5]  = a[5] & b[5];  assign caxb[5] = c[5]   & axb[5];  assign c[6] = anb[5] | caxb[5];
 
    assign axb[6]  = a[6] ^ b[6];  assign sum[6]  = axb[6] ^ c[6];
    assign anb[6]  = a[6] & b[6];  assign caxb[6] = c[6]   & axb[6];  assign c[7] = anb[6] | caxb[6];
 
    assign axb[7]  = a[7] ^ b[7];  assign sum[7]  = axb[7] ^ c[7];
    assign anb[7]  = a[7] & b[7];  assign caxb[7] = c[7]   & axb[7];  assign c[8] = anb[7] | caxb[7];
    // c[8] (carry out) unused
 
endmodule