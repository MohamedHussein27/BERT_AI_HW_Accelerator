module scale_rom #(
    parameter ADDR_WIDTH = 8
)(
    // Address format: {layer_id[3:0], block_id[3:0]}
    input  logic [ADDR_WIDTH-1:0]  addr,
    
    output logic [31:0]            m0_out,
    output logic signed [7:0]      s_out
);

    // Synthesis pragma to explicitly force LUT usage (Distributed ROM) instead of BRAM
    // (* rom_style = "distributed" *) 

    always_comb begin
        case (addr)
            // ================= LAYER 0 =================
            8'h00: begin m0_out = 32'd1972133394; s_out =  8'sd11; end // 0: ATTEN-Q
            8'h01: begin m0_out = 32'd1104008503; s_out =  8'sd10; end // 1: ATTEN-K
            8'h02: begin m0_out = 32'd1958868565; s_out =  8'sd10; end // 2: ATTEN-V
            8'h03: begin m0_out = 32'd2051840830; s_out = -8'sd12; end // 3: QKt Dequant
            8'h04: begin m0_out = 32'd1417471555; s_out =  8'sd5;  end // 4: Softmax Quant
            8'h05: begin m0_out = 32'd2130779236; s_out =  8'sd8;  end // 5: .VGeMM Quant
            8'h06: begin m0_out = 32'd1614228903; s_out =  8'sd10; end // 6: Wo Quantize
            8'h07: begin m0_out = 32'd1845025262; s_out =  8'sd21; end // 7: Add&Norm Quant
            8'h08: begin m0_out = 32'd1449841874; s_out = -8'sd6;  end // 8: FFN1 Dequant
            8'h09: begin m0_out = 32'd1556097776; s_out =  8'sd10; end // 9: GELU Quant
            8'h0A: begin m0_out = 32'd1763007950; s_out =  8'sd11; end // 10: FFN2 Quant

            // ================= LAYER 1 =================
            8'h10: begin m0_out = 32'd1081081080; s_out =  8'sd10; end 
            8'h11: begin m0_out = 32'd1108794819; s_out =  8'sd10; end 
            8'h12: begin m0_out = 32'd1393456876; s_out =  8'sd10; end 
            8'h13: begin m0_out = 32'd1963391527; s_out = -8'sd12; end 
            8'h14: begin m0_out = 32'd1193079389; s_out =  8'sd5;  end 
            8'h15: begin m0_out = 32'd1571021856; s_out =  8'sd7;  end 
            8'h16: begin m0_out = 32'd1656627290; s_out =  8'sd11; end 
            8'h17: begin m0_out = 32'd1783102513; s_out =  8'sd21; end 
            8'h18: begin m0_out = 32'd1500191834; s_out = -8'sd6;  end 
            8'h19: begin m0_out = 32'd1533405172; s_out =  8'sd10; end 
            8'h1A: begin m0_out = 32'd1860967253; s_out =  8'sd11; end 

            // ================= LAYER 2 =================
            8'h20: begin m0_out = 32'd1138621866; s_out =  8'sd10; end 
            8'h21: begin m0_out = 32'd1093909335; s_out =  8'sd10; end 
            8'h22: begin m0_out = 32'd1221084528; s_out =  8'sd10; end 
            8'h23: begin m0_out = 32'd1967157660; s_out = -8'sd12; end 
            8'h24: begin m0_out = 32'd1243924371; s_out =  8'sd5;  end 
            8'h25: begin m0_out = 32'd1759490783; s_out =  8'sd8;  end 
            8'h26: begin m0_out = 32'd1298907977; s_out =  8'sd11; end 
            8'h27: begin m0_out = 32'd1753598033; s_out =  8'sd21; end 
            8'h28: begin m0_out = 32'd1525432187; s_out = -8'sd6;  end 
            8'h29: begin m0_out = 32'd1467772199; s_out =  8'sd10; end 
            8'h2A: begin m0_out = 32'd1658178314; s_out =  8'sd11; end 

            // ================= LAYER 3 =================
            8'h30: begin m0_out = 32'd1089659065; s_out =  8'sd10; end 
            8'h31: begin m0_out = 32'd1119828917; s_out =  8'sd10; end 
            8'h32: begin m0_out = 32'd1218363151; s_out =  8'sd10; end 
            8'h33: begin m0_out = 32'd2109976621; s_out = -8'sd12; end 
            8'h34: begin m0_out = 32'd1302981916; s_out =  8'sd5;  end 
            8'h35: begin m0_out = 32'd1696809140; s_out =  8'sd8;  end 
            8'h36: begin m0_out = 32'd1640903450; s_out =  8'sd11; end 
            8'h37: begin m0_out = 32'd1672594262; s_out =  8'sd21; end 
            8'h38: begin m0_out = 32'd1599308896; s_out = -8'sd6;  end 
            8'h39: begin m0_out = 32'd1537999004; s_out =  8'sd10; end 
            8'h3A: begin m0_out = 32'd1162848325; s_out =  8'sd11; end 

            // ================= LAYER 4 =================
            8'h40: begin m0_out = 32'd1135749098; s_out =  8'sd10; end 
            8'h41: begin m0_out = 32'd1114481011; s_out =  8'sd10; end 
            8'h42: begin m0_out = 32'd1351187335; s_out =  8'sd10; end 
            8'h43: begin m0_out = 32'd1098649998; s_out = -8'sd13; end // Note: Shift is -13 here
            8'h44: begin m0_out = 32'd1390594902; s_out =  8'sd5;  end 
            8'h45: begin m0_out = 32'd1512890788; s_out =  8'sd8;  end 
            8'h46: begin m0_out = 32'd1679396499; s_out =  8'sd11; end 
            8'h47: begin m0_out = 32'd1643657618; s_out =  8'sd21; end 
            8'h48: begin m0_out = 32'd1627465355; s_out = -8'sd6;  end 
            8'h49: begin m0_out = 32'd1517408184; s_out =  8'sd10; end 
            8'h4A: begin m0_out = 32'd1325926245; s_out =  8'sd11; end 

            // ================= LAYER 5 =================
            8'h50: begin m0_out = 32'd1248588427; s_out =  8'sd10; end 
            8'h51: begin m0_out = 32'd1272097990; s_out =  8'sd10; end 
            8'h52: begin m0_out = 32'd1479218190; s_out =  8'sd10; end 
            8'h53: begin m0_out = 32'd2031116760; s_out = -8'sd12; end 
            8'h54: begin m0_out = 32'd1436132024; s_out =  8'sd5;  end 
            8'h55: begin m0_out = 32'd1459120748; s_out =  8'sd8;  end 
            8'h56: begin m0_out = 32'd1875472705; s_out =  8'sd11; end 
            8'h57: begin m0_out = 32'd1548415368; s_out =  8'sd21; end 
            8'h58: begin m0_out = 32'd1727569319; s_out = -8'sd6;  end 
            8'h59: begin m0_out = 32'd1477838087; s_out =  8'sd10; end 
            8'h5A: begin m0_out = 32'd1447181179; s_out =  8'sd11; end 

            // ================= LAYER 6 =================
            8'h60: begin m0_out = 32'd1214810854; s_out =  8'sd10; end 
            8'h61: begin m0_out = 32'd1272449740; s_out =  8'sd10; end 
            8'h62: begin m0_out = 32'd1471063231; s_out =  8'sd10; end 
            8'h63: begin m0_out = 32'd2046075739; s_out = -8'sd12; end 
            8'h64: begin m0_out = 32'd1384857480; s_out =  8'sd5;  end 
            8'h65: begin m0_out = 32'd1570281119; s_out =  8'sd8;  end 
            8'h66: begin m0_out = 32'd1754558567; s_out =  8'sd11; end 
            8'h67: begin m0_out = 32'd1484694599; s_out =  8'sd21; end 
            8'h68: begin m0_out = 32'd1801713891; s_out = -8'sd6;  end 
            8'h69: begin m0_out = 32'd1418623495; s_out =  8'sd10; end 
            8'h6A: begin m0_out = 32'd1098297178; s_out =  8'sd11; end 

            // ================= LAYER 7 =================
            8'h70: begin m0_out = 32'd1296299044; s_out =  8'sd10; end 
            8'h71: begin m0_out = 32'd1291165488; s_out =  8'sd10; end 
            8'h72: begin m0_out = 32'd1447246344; s_out =  8'sd10; end 
            8'h73: begin m0_out = 32'd1943915767; s_out = -8'sd12; end 
            8'h74: begin m0_out = 32'd1464815885; s_out =  8'sd5;  end 
            8'h75: begin m0_out = 32'd1395894581; s_out =  8'sd8;  end 
            8'h76: begin m0_out = 32'd2084748259; s_out =  8'sd11; end 
            8'h77: begin m0_out = 32'd1515197706; s_out =  8'sd21; end 
            8'h78: begin m0_out = 32'd1765443425; s_out = -8'sd6;  end 
            8'h79: begin m0_out = 32'd1483742322; s_out =  8'sd10; end 
            8'h7A: begin m0_out = 32'd1242701252; s_out =  8'sd11; end 

            // ================= LAYER 8 =================
            8'h80: begin m0_out = 32'd1376712388; s_out =  8'sd10; end 
            8'h81: begin m0_out = 32'd1352191951; s_out =  8'sd10; end 
            8'h82: begin m0_out = 32'd1588220302; s_out =  8'sd10; end 
            8'h83: begin m0_out = 32'd1944943637; s_out = -8'sd12; end 
            8'h84: begin m0_out = 32'd1310416147; s_out =  8'sd5;  end 
            8'h85: begin m0_out = 32'd1415542743; s_out =  8'sd8;  end 
            8'h86: begin m0_out = 32'd2055817848; s_out =  8'sd11; end 
            8'h87: begin m0_out = 32'd1481095769; s_out =  8'sd21; end 
            8'h88: begin m0_out = 32'd1806092411; s_out = -8'sd6;  end 
            8'h89: begin m0_out = 32'd1522039072; s_out =  8'sd10; end 
            8'h8A: begin m0_out = 32'd1240877525; s_out =  8'sd11; end 

            // ================= LAYER 9 =================
            8'h90: begin m0_out = 32'd1328041062; s_out =  8'sd10; end 
            8'h91: begin m0_out = 32'd1313099036; s_out =  8'sd10; end 
            8'h92: begin m0_out = 32'd1518650030; s_out =  8'sd10; end 
            8'h93: begin m0_out = 32'd1917921658; s_out = -8'sd12; end 
            8'h94: begin m0_out = 32'd1506278218; s_out =  8'sd5;  end 
            8'h95: begin m0_out = 32'd1463116553; s_out =  8'sd8;  end 
            8'h96: begin m0_out = 32'd1577039736; s_out =  8'sd11; end 
            8'h97: begin m0_out = 32'd1481596062; s_out =  8'sd21; end 
            8'h98: begin m0_out = 32'd1805482544; s_out = -8'sd6;  end 
            8'h99: begin m0_out = 32'd1511626363; s_out =  8'sd10; end 
            8'h9A: begin m0_out = 32'd1087376496; s_out =  8'sd11; end 

            // ================= LAYER 10 =================
            8'hA0: begin m0_out = 32'd1342858781; s_out =  8'sd10; end 
            8'hA1: begin m0_out = 32'd1296595168; s_out =  8'sd10; end 
            8'hA2: begin m0_out = 32'd1461924128; s_out =  8'sd10; end 
            8'hA3: begin m0_out = 32'd1966700741; s_out = -8'sd12; end 
            8'hA4: begin m0_out = 32'd1529508551; s_out =  8'sd5;  end 
            8'hA5: begin m0_out = 32'd1347887107; s_out =  8'sd8;  end 
            8'hA6: begin m0_out = 32'd2106201677; s_out =  8'sd11; end 
            8'hA7: begin m0_out = 32'd1514610199; s_out =  8'sd21; end 
            8'hA8: begin m0_out = 32'd1766125111; s_out = -8'sd6;  end 
            8'hA9: begin m0_out = 32'd1567838832; s_out =  8'sd10; end 
            8'hAA: begin m0_out = 32'd1968255041; s_out =  8'sd12; end // Note: Shift is 12 here

            // ================= LAYER 11 =================
            8'hB0: begin m0_out = 32'd1421861524; s_out =  8'sd10; end 
            8'hB1: begin m0_out = 32'd1430590577; s_out =  8'sd10; end 
            8'hB2: begin m0_out = 32'd1359846296; s_out =  8'sd10; end 
            8'hB3: begin m0_out = 32'd1832810226; s_out = -8'sd12; end 
            8'hB4: begin m0_out = 32'd1510971945; s_out =  8'sd5;  end 
            8'hB5: begin m0_out = 32'd1432523331; s_out =  8'sd8;  end 
            8'hB6: begin m0_out = 32'd1155174891; s_out =  8'sd10; end 
            8'hB7: begin m0_out = 32'd1395082723; s_out =  8'sd21; end 
            8'hB8: begin m0_out = 32'd1917446029; s_out = -8'sd6;  end 
            8'hB9: begin m0_out = 32'd1514380517; s_out =  8'sd10; end 
            8'hBA: begin m0_out = 32'd1753913888; s_out =  8'sd12; end 

            // Default fallback to safely prevent latches
            default: begin 
                m0_out = 32'd0; 
                s_out  = 8'sd0; 
            end
        endcase
    end
endmodule