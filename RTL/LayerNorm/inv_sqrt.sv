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
    localparam SHIFT_WIDTH = $clog2(DATAWIDTH) + 1;;
    localparam MUL_WIDTH   = DATAWIDTH * 2;
    // 3.0 in Q5.26 format
    localparam logic signed [DATAWIDTH-1:0] THREE_Q5_26 = signed'(DATAWIDTH'(3) <<<FRAC_BITS);
    

    // FSM Definition
    typedef enum logic [3:0] {
        IDLE,
        LOD,
        NORMALIZE,
        LUT_INIT,
        CALC_X2,
        CALC_AX2,
        SUBTRACT,
        CALC_FINAL,
        ITER_CHECK,
        DENORMALIZE
    } state_t;

    state_t cs;

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
    logic signed [DATAWIDTH-1:0] sub_res;
    logic signed [DATAWIDTH-1:0] final_res;

    // Mux for iteration
    logic signed [DATAWIDTH-1:0] x_current;
    assign x_current = (iter_cnt == 0) ? x0 : xn;

    // Main FSM Logic
    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            cs <= IDLE;
            error <= 0;
            busy <= 0;
            valid_out <= 0;
            shift_val <= 0;
            lut_addr <= 0;
            iter_cnt <= 0;
            data_out <= 0;
            a_reg <= 0;
            a_norm <= 0;
            xn <= 0;
            x2 <= 0;
            ax2 <= 0;
            sub_res <= 0;
            final_res <= 0;
            half_shift <= 0;
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
                        // Division by zero guard
                        if (data_in <= 0) begin
                            data_out <= 0; 
                            error <= 1;
                            cs <= IDLE;
                        end else begin
                            busy <= 1;
                            cs <= LOD;
                        end
                    end
                end

                LOD: begin
                    leading_one(a_reg, shift_val);
                    cs <= NORMALIZE;
                end

                NORMALIZE: begin
                    // Perform range reduction to [0.5, 2.0)
                    if (shift_val >= 0)
                        a_norm <= a_reg <<< shift_val;
                    else
                        a_norm <= a_reg >>> (-shift_val);

                    // Force arithmetic shift for the signed half_shift
                    half_shift <= shift_val >>> 1;
                    cs <= LUT_INIT; 
                end

                LUT_INIT: begin
                    lut_addr <= a_norm[FRAC_BITS:FRAC_BITS-LUT_ADDR+1]; 
                    cs <= CALC_X2;
                end

                CALC_X2: begin
                    // Compute X^2 and align back to Q5.26
                    x2 <= (MUL_WIDTH'(x_current) * x_current) >>> FRAC_BITS;
                    cs <= CALC_AX2;
                end

                CALC_AX2: begin
                    // Compute a * X^2
                    ax2 <= (MUL_WIDTH'(a_norm) * x2) >>> FRAC_BITS;
                    cs <= SUBTRACT;
                end

                SUBTRACT: begin
                    // Compute (3.0 - aX^2)
                    sub_res <= THREE_Q5_26 - ax2;
                    cs <= CALC_FINAL;
                end

                CALC_FINAL: begin
                    // Compute (X/2) * (3.0 - aX^2)
                    final_res <= (MUL_WIDTH'(x_current >>> 1) * sub_res) >>> FRAC_BITS;
                    cs <= ITER_CHECK;
                end

                ITER_CHECK: begin
                    if (iter_cnt == 1) begin
                        cs <= DENORMALIZE;
                    end else begin
                        iter_cnt <= iter_cnt + 1;
                        xn <= final_res;
                        cs <= CALC_X2; // Loop back for 2nd iteration
                    end
                end

                DENORMALIZE: begin
                    // Reverse the normalization using the square root property
                    if (half_shift > 0)
                        data_out <= final_res <<< half_shift; 
                    else if (half_shift < 0)
                        data_out <= final_res >>> (-half_shift); 
                    else
                        data_out <= final_res;

                    valid_out <= 1;
                    busy <= 0;
                    cs <= IDLE;
                end
                default: cs <= IDLE;
            endcase
        end
    end

    //  Leading One Detector (LOD) Function
    function void leading_one (
        input  logic signed [DATAWIDTH-1:0] val,
        output logic signed [SHIFT_WIDTH-1:0] shift
    );
        shift = 0;
        for (int i = DATAWIDTH-1; i >= 0; i--) begin
            if (val[i]) begin
                shift = $signed(FRAC_BITS - i);
                // Force even shift to keep sqrt adjustment as a power of 2
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