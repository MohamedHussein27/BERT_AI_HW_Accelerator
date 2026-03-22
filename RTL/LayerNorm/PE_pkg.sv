package PE_pkg;
    // Opcodes defining the PE Mux routing
    typedef enum logic [2:0] {
        OP_LOAD_WGT   = 3'd0, // Load Gamma into local mem
        OP_LOAD_BIAS  = 3'd1, // Load Beta into local mem
        OP_PASS_X     = 3'd2, // Output X (For Mean Accumulation)
        OP_VAR_SQR    = 3'd3, // Output (X - mu)^2
        OP_NORMALIZE  = 3'd4, // Output (X - mu) * inv_sigma
        OP_AFFINE     = 3'd5, // Output (X_norm * gamma) + beta
        OP_LOAD_MEAN  = 3'd6  // to load mean when we need it in normalization
    } pe_op_e;
endpackage