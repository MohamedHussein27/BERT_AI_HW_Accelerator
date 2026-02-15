module systolic_controller #(
    parameter DATAWIDTH = 8,
    parameter N_SIZE = 32,
    parameter num_of_raws = 512,
    parameter BUS_WIDTH = 256, // N_SIZE * DATAWIDTH
    parameter ADDR_WIDTH  = 10
) (
    input  logic clk,
    input  logic rst_n,
    input  logic valid_in, // indicated a valid input
    input  logic load_weight, // should be high as long as there is a weight is being fetched.
    // Control signals to the Systolic Array
    output logic sys_wt_en,
    output logic we,
    output logic [ADDR_WIDTH-1:0] rd_addr,
    output logic [ADDR_WIDTH-1:0] wr_addr,
    
    // Status signals
    output logic ready,
    output logic busy,
    output logic done
);  

    localparam [1:0] IDLE = 2'b00;
    localparam [1:0] LOADING_WEIGHT = 2'b01;
    localparam [1:0] COMPUTE = 2'b10;

    localparam count_bits = $clog2(num_of_raws + N_SIZE - 1);
    localparam wt_count_bits = $clog2(N_SIZE);

    logic [1:0] cs ,ns;
    logic [count_bits-1:0] cycle_cnt;
    logic [wt_count_bits-1:0] wt_load_cnt;

    // assign current state to next state
    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) cs <= IDLE;
        else        cs <= ns;
    end

    // next state logic
    always @(*) begin
        case (cs)
            IDLE: begin
                if (load_weight)
                    ns = LOADING_WEIGHT;
                else if (valid_in)
                    ns = COMPUTE;
            end
            
            LOADING_WEIGHT: begin
                if ((!load_weight) || (wt_load_cnt == N_SIZE-1))begin
                    ns = IDLE;
                end
            end
            
            COMPUTE: begin
                if (cycle_cnt == num_of_raws + 2*N_SIZE - 2)
                    ns = IDLE;
            end     
            default: ns = IDLE;
        endcase
    end
    
    // output logic
    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            busy <= 0;
            done <= 0;
            ready <= 1;
            we <= 0;
            sys_wt_en  <= 0;
            wr_addr <= 0;
            rd_addr <= 0;
        end
        else begin
            case (cs)
                IDLE: begin
                    sys_wt_en <= 0;
                    wr_addr <= 0;
                    rd_addr <= 0;
                    we <= 0;
                    done <= 0;
                    busy <= 0;
                    ready <= 1;
                end

                LOADING_WEIGHT: begin
                    sys_wt_en <= 1;
                    wr_addr <= 0;
                    rd_addr <= 0;
                    we <= 0;
                    done <= 0;
                    busy <= 1;
                    ready <= 0;
                end 

                COMPUTE: begin
                    sys_wt_en <= 0;
                    if (cycle_cnt < num_of_raws + 2*N_SIZE - 2) begin
                        rd_addr <= rd_addr + 1;
                    end
                    if ((cycle_cnt >= N_SIZE - 1) && (cycle_cnt < num_of_raws + 2*N_SIZE - 2))begin 
                        we <= 1; 
                        wr_addr <= wr_addr + 1;
                    end 
                    if (cycle_cnt == num_of_raws + 2*N_SIZE - 2) done <= 1; 
                    busy <= 1;
                    ready <= 0;
                end  
            endcase
        end
    end


    // counter circuit 
    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            cycle_cnt <= '0;
        end
        else if (cs == COMPUTE && valid_in) begin
            cycle_cnt <= cycle_cnt + 1;
        end
        else if (cs == IDLE) begin
            cycle_cnt <= '0;
        end
    end

    // Weight loading counter
    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            wt_load_cnt <= '0;
        end
        else if (cs == LOADING_WEIGHT) begin
            wt_load_cnt <= wt_load_cnt + 1;
        end
        else begin
            wt_load_cnt <= '0;
        end
    end
endmodule