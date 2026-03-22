module inv_sqrt #(
    parameter DATAWIDTH = 32,
    parameter FRAC_BITS = 26,
    parameter LUT_ADDR = 4
) (
    input  logic clk,
    input  logic rst_n,
    input  logic valid_in,
    input  logic signed [DATAWIDTH-1:0] data_in, // Q5.26 format

    output logic signed [DATAWIDTH-1:0] data_out,
    output logic valid_out,
    output logic error,
    output logic busy
);

    // Parameters & Constants
    localparam SHIFT_WIDTH = $clog2(DATAWIDTH) + 1;
    localparam MUL_WIDTH   = DATAWIDTH * 2;
    localparam logic signed [DATAWIDTH-1:0] THREE_Q5_26 = signed'(DATAWIDTH'(3) <<< FRAC_BITS);

    // FSM Definition
    typedef enum logic [2:0] {
        IDLE,
        LOD_NORM,
        CALC_X2,
        CALC_AX2,
        CALC_FINAL,
        ITER_CHECK
    } state_t;

    state_t cs, ns;

    // Internal Regs
    logic signed [DATAWIDTH-1:0] a_reg;
    logic signed [DATAWIDTH-1:0] a_norm;
    logic signed [DATAWIDTH-1:0] xn;
    logic signed [DATAWIDTH-1:0] x0;
    
    // Control
    logic signed [SHIFT_WIDTH-1:0] shift_val;
    logic signed [SHIFT_WIDTH-1:0] half_shift;
    logic [1:0] iter_cnt;
    logic [3:0] lut_addr;

    // Calculation Pipeline Regs
    logic signed [DATAWIDTH-1:0] x2;
    logic signed [DATAWIDTH-1:0] ax2;
    logic signed [DATAWIDTH-1:0] final_res;

    // Mux for iteration
    logic signed [DATAWIDTH-1:0] x_current;
    assign x_current = (iter_cnt == 0) ? x0 : xn;

    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) cs <= IDLE;
        else        cs <= ns;
    end

    // NEXT STATE LOGIC
    always_comb begin
        case (cs)
            IDLE: begin
                if (valid_in) begin
                    if (data_in <= 0) ns = IDLE;
                    else              ns = LOD_NORM;
                end
            end
            LOD_NORM: begin
                    leading_one(a_reg, shift_val);
                    if (shift_val >= 0)
                        a_norm = a_reg <<< shift_val;
                    else
                        a_norm = a_reg >>> (-shift_val);

                    half_shift = shift_val >>> 1;
                    lut_addr = a_norm[FRAC_BITS:FRAC_BITS-LUT_ADDR+1]; 

                    ns = CALC_X2;
            end
            CALC_X2:    ns = CALC_AX2;
            CALC_AX2:   ns = CALC_FINAL;
            CALC_FINAL: begin
                    if (iter_cnt == 1) ns = IDLE;
                    else               ns = CALC_X2;
            end
            
            default:    ns = IDLE;
        endcase
    end

    // OUTPUT LOGIC
    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            error <= 0;
            busy <= 0;
            valid_out <= 0;
            iter_cnt <= 0;
            data_out <= 0;
            a_reg <= 0;
            xn <= 0;
            x2 <= 0;
            ax2 <= 0;
            final_res <= 0;
        end
        else begin
            case (cs)
                IDLE: begin
                    valid_out <= 0;
                    error <= 0;
                    busy <= 0;
                    iter_cnt <= 0;
                    if (valid_in) begin
                        a_reg <= data_in;
                        if (data_in <= 0) begin
                            data_out <= 0; 
                            error <= 1;
                        end else begin
                            busy <= 1;
                        end
                    end
                end

                LOD_NORM: begin
                    // comp logic
                end

                CALC_X2: begin
                    x2 <= (MUL_WIDTH'(x_current) * x_current) >>> FRAC_BITS;
                end

                CALC_AX2: begin
                    ax2 <= THREE_Q5_26 - ((MUL_WIDTH'(a_norm) * x2) >>> FRAC_BITS);
                end

                CALC_FINAL: begin
                    if (iter_cnt == 1) begin
                        if (half_shift > 0)
                            data_out <= ((MUL_WIDTH'(x_current >>> 1) * ax2) >>> FRAC_BITS) <<< half_shift; 
                        else if (half_shift < 0)
                            data_out <= ((MUL_WIDTH'(x_current >>> 1) * ax2) >>> FRAC_BITS) >>> (-half_shift); 
                        else
                            data_out <= ((MUL_WIDTH'(x_current >>> 1) * ax2) >>> FRAC_BITS);

                        valid_out <= 1;
                        busy <= 0;
                    end else begin
                        iter_cnt <= iter_cnt + 1;
                        xn <= (MUL_WIDTH'(x_current >>> 1) * ax2) >>> FRAC_BITS;
                    end
                end

            endcase
        end
    end

    // Leading One Detector (LOD) Function
    function void leading_one (
        input  logic signed [DATAWIDTH-1:0] val,
        output logic signed [SHIFT_WIDTH-1:0] shift
    );
        shift = 0;
        for (int i = DATAWIDTH-1; i >= 0; i--) begin
            if (val[i]) begin
                shift = $signed(FRAC_BITS - i);
                if (shift[0] != 0)
                    shift = shift - 1;
                break;
            end
        end
    endfunction

    // LUT Instantiation
    inv_sqrt_lut LUT (
        .index(lut_addr),
        .x0(x0)
    );
endmodule