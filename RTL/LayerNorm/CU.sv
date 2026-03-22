module layer_norm_fsm #(

) (

);

    // FSM
    typedef enum logic [2:0] {
        IDLE,
        LN_MEAN,     // calc mean
        LN_VAR,      // calc variance
        LN_STD_NORM, // calc 1/root(var)
        LN_WGT,      // multibly by weights
        LN_BIAS      // add bias
    } state_t;

    state_t cs, ns;


    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n)
            cs <= IDLE;
        else
            cs <= ns;
    end

    // Next state logic
    always_comb begin
        case (cs)
            IDLE    : begin
                
            end
            LN_MEAN : begin
                
            end
            LN_VAR  :begin
                
            end 
            LN_STD  : begin
                
            end
            LN_NORM : begin
                
            end 
            LN_WGT  : begin
                
            end
            LN_BIAS : begin
                
            end
            default :
        endcase
    end

    // Output logic
    always_comb begin

        case (cs)
            IDLE    : begin
                
            end
            LN_MEAN : begin
                
            end
            LN_VAR  :begin
                
            end 
            LN_STD  : begin
                
            end
            LN_NORM : begin
                
            end 
            LN_WGT  : begin
                
            end
            LN_BIAS : begin
                
            end
            default :
        endcase
    end
endmodule