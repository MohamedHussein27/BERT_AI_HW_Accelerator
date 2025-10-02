import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import numpy as np

# ==============================================================================
# Shared Hardware Components (EU and DU)
# ==============================================================================

class ExponentialUnit(nn.Module):
    """
    Exponential Unit (EU) - Shared between SCU and GCU
    Hardware-friendly base-2 exponential computation using piecewise linear approximation
    Based on the paper's equation (10): 2^x = 2^frac(x) << int(x)
    """
    def __init__(self, num_segments=8):
        super().__init__()
        self.num_segments = num_segments
        
        # Pre-compute piecewise linear coefficients for 2^frac(x) approximation
        # Using Ki and Bi values as mentioned in the paper's LUT approach
        coeffs = []
        for i in range(num_segments):
            x_start = i / num_segments
            x_end = (i + 1) / num_segments
            
            # Linear approximation: y = K*x + B for 2^x in [0,1]
            y_start = 2 ** x_start
            y_end = 2 ** x_end
            K = (y_end - y_start) / (x_end - x_start)
            B = y_start - K * x_start
            
            coeffs.append([K, B])
        
        self.register_buffer('coefficients', torch.tensor(coeffs, dtype=torch.float32))
    
    def forward(self, x, use_log2e_scaling=True):
        """
        Compute 2^x using piecewise linear approximation
        
        Args:
            x: Input tensor
            use_log2e_scaling: If True, converts e^x to 2^(log2(e)*x) for SCU
                              If False, computes 2^x directly for GCU division
        """
        if use_log2e_scaling:
            # Paper: "exi is equal to 2^(log2e·xi)"
            # log2(e) ≈ 1.0111 in binary ≈ 1 + 0.1 - 0.0001 (paper's approximation)
            log2e = 1.0 + 0.5 - 0.0625  # Hardware-friendly approximation using shifts
            x = x * log2e
        
        # Split into integer and fractional parts (equation 10)
        x_int = torch.floor(x).long()
        x_frac = x - x_int.float()
        
        # Clamp fractional part to [0, 1) for lookup table
        x_frac = torch.clamp(x_frac, 0, 0.999999)
        
        # Find which segment each fractional part belongs to
        segment_idx = torch.floor(x_frac * self.num_segments).long()
        segment_idx = torch.clamp(segment_idx, 0, self.num_segments - 1)
        
        # Get coefficients Ki and Bi for each segment
        K = self.coefficients[segment_idx, 0]
        B = self.coefficients[segment_idx, 1]
        
        # Apply piecewise linear approximation: 2^frac(x) = K*x + B
        frac_result = K * x_frac + B
        
        # Apply integer part as bit shifts: 2^int(x) * 2^frac(x)
        # Paper: "2^frac(xi) << int(xi)" where << denotes shift operation
        x_int_clamped = torch.clamp(x_int, -15, 15)  # Prevent overflow
        result = frac_result * (2.0 ** x_int_clamped.float())
        
        return result

class DivisionUnit(nn.Module):
    """
    Division Unit (DU) - Shared between SCU and GCU
    Hardware-friendly division using equations (11-12) from the paper:
    F1/F2 = 2^((m1+w1)-(m2+w2))
    """
    def __init__(self):
        super().__init__()
        self.eu = ExponentialUnit()
    
    def leading_one_detector(self, x):
        """
        Simulates Leading One Detector (LOD) to find w and m values
        Paper: "F1 = w1 × 2^m1, F2 = w2 × 2^m2 where m1, m2 ∈ [1,2)"
        """
        abs_x = torch.abs(x)
        abs_x = torch.clamp(abs_x, min=1e-8)  # Avoid log of zero
        
        # Find the position of leading 1 bit
        log2_x = torch.log2(abs_x)
        w = torch.floor(log2_x).long()  # Integer part
        m_minus_1 = log2_x - w.float()  # Fractional part
        m = m_minus_1 + 1.0  # Ensure m ∈ [1,2)
        
        return w, m

    def forward(self, numerator, denominator, add_one_to_denominator=False):
        """
        Perform division using the paper's method
        
        Args:
            numerator: F1 in the paper
            denominator: F2 in the paper  
            add_one_to_denominator: For GCU, adds 1 to denominator as per equation (8)
        """
        # Handle the GCU case: xi / (1 + 2^(-2*log2e*h(xi)))
        if add_one_to_denominator:
            denominator = 1.0 + denominator
        
        # Get LOD values for numerator and denominator
        w1, m1 = self.leading_one_detector(numerator)
        w2, m2 = self.leading_one_detector(denominator)
        
        # Apply equation (12): F1/F2 = 2^((m1+w1)-(m2+w2))
        exponent = (m1 + w1.float()) - (m2 + w2.float())
        
        # Use exponential unit to compute 2^exponent (no log2e scaling for division)
        result = self.eu(exponent, use_log2e_scaling=False)
        
        # Handle signs
        sign = torch.sign(numerator) * torch.sign(denominator)
        result = result * sign
        
        return result

# ==============================================================================
# SCU-Specific Components
# ==============================================================================

class FindMaxUnit(nn.Module):
    """
    Find Max Unit (FMU) for SCU
    Hardware-friendly maximum finding using parallel comparison tree
    Paper: "O(⌈log2 n⌉) complexity instead of O(n)"
    """
    def __init__(self):
        super().__init__()
    
    def forward(self, x):
        """
        Find maximum value along the last dimension using tree-based reduction
        Simulates the parallel comparison groups from Figure 7
        """
        # The paper mentions grouping (x0,x1,...,x31) as Group 1, (x32,...,x47) as Group 2, etc.
        # We simulate this with efficient torch operations
        max_vals, _ = torch.max(x, dim=-1, keepdim=True)
        return max_vals

class AdderTree(nn.Module):
    """
    Adder Tree for SCU - Parallel summation of exponential values
    """
    def __init__(self):
        super().__init__()
    
    def forward(self, x):
        """Sum along the last dimension using tree-based reduction"""
        return torch.sum(x, dim=-1, keepdim=True)

# ==============================================================================
# GCU-Specific Components  
# ==============================================================================

class PolynomialUnit(nn.Module):
    """
    Polynomial computation unit for GCU
    Computes h(xi) = sqrt(2/π) * xi + 0.044715 * xi^3 from equation (7)
    Using hardware-friendly approximations from equation (9)
    """
    def __init__(self):
        super().__init__()
        
        # Hardware-friendly coefficient approximations from the paper
        # sqrt(2/π) ≈ 0.7978... 
        # Paper uses binary approximation for hardware efficiency
        self.sqrt_2_over_pi = 0.8  # Simplified for hardware
        
        # 0.044715 ≈ 0.000011 in binary = 0.000001 + 0.00001 (paper's approximation)
        self.cubic_coeff = 0.03125 + 0.03125  # Using shift operations: 1/32 + 1/32
    
    def forward(self, x):
        """
        Compute h(x) = sqrt(2/π) * x + 0.044715 * x^3
        Paper equation (9): s(xi) = -2*log2(e) * h(xi)
        """
        linear_term = self.sqrt_2_over_pi * x
        cubic_term = self.cubic_coeff * (x ** 3)
        h_x = linear_term + cubic_term
        
        # Apply the coefficient from equation (9)
        # -2*log2(e)*sqrt(2/π) ≈ -10.0101 = -10 - 0.01 - 0.0001 (paper's binary approximation)
        coeff = -10.0 - 0.25 - 0.0625  # Hardware-friendly using shifts
        s_x = coeff * h_x
        
        return s_x

# ==============================================================================
# Main SCU and GCU Classes
# ==============================================================================

class SCU(nn.Module):
    """
    Softmax Compute Unit (SCU) - Hardware-friendly softmax implementation
    Based on Figure 6 from the paper with four stages:
    Stage 1: Find Max Unit (FMU)
    Stage 2: Exponential Unit (EU) 
    Stage 3: Adder Tree
    Stage 4: Division Unit (DU)
    """
    def __init__(self, use_mask=False):
        super().__init__()
        self.use_mask = use_mask
        
        # Initialize submodules according to paper architecture
        self.fmu = FindMaxUnit()          # Stage 1: Find maximum
        self.eu = ExponentialUnit()       # Stage 2: Exponential computation  
        self.adder_tree = AdderTree()     # Stage 3: Sum computation
        self.du = DivisionUnit()          # Stage 4: Division computation
    
    def forward(self, scores, attention_mask=None):
        """
        Complete SCU dataflow implementing equation (6):
        f(xi) = 2^(log2e(xi-xmax)) / Σ(2^(log2e(xj-xmax)))
        """
        # Apply attention mask if provided
        if attention_mask is not None and self.use_mask:
            scores = scores + attention_mask
        
        # Stage 1: Find maximum value xmax (FMU)
        x_max = self.fmu(scores)
        
        # Stage 2: Compute xi - xmax and apply EU for exponential
        shifted_scores = scores - x_max  # Numerical stability
        exp_scores = self.eu(shifted_scores, use_log2e_scaling=True)
        
        # Stage 3: Sum all exponentials (Adder Tree)
        sum_exp = self.adder_tree(exp_scores)
        
        # Stage 4: Division exp_scores / sum_exp (DU)
        probabilities = self.du(exp_scores, sum_exp, add_one_to_denominator=False)
        
        # Ensure numerical stability and proper normalization
        probabilities = torch.clamp(probabilities, min=1e-8, max=1.0)
        prob_sum = torch.sum(probabilities, dim=-1, keepdim=True)
        probabilities = probabilities / (prob_sum + 1e-8)
        
        return probabilities

class GCU(nn.Module):
    """
    GELU Compute Unit (GCU) - Hardware-friendly GELU implementation
    Based on Figure 10 from the paper with four stages:
    Stage 1: Polynomial computation h(xi)
    Stage 2: Exponential Unit (EU) for 2^(-2*log2e*h(xi))
    Stage 3: Division Unit (DU) for xi/(1 + 2^(-2*log2e*h(xi)))
    Stage 4: Final EU processing (if needed)
    """
    def __init__(self):
        super().__init__()
        
        # Initialize submodules according to paper architecture
        self.polynomial_unit = PolynomialUnit()  # Stage 1: Compute h(xi)
        self.eu = ExponentialUnit()              # Stage 2: Exponential computation
        self.du = DivisionUnit()                 # Stage 3: Division computation
    
    def forward(self, x):
        """
        Complete GCU dataflow implementing equation (8):
        g(xi) = xi / (1 + 2^(-2*log2e*h(xi)))
        where h(xi) = sqrt(2/π)*xi + 0.044715*xi^3
        """
        # Stage 1: Polynomial computation h(xi)
        s_x = self.polynomial_unit(x)  # This gives us -2*log2(e)*h(xi)
        
        # Stage 2: Exponential computation 2^(-2*log2e*h(xi))
        # Since s_x already includes the -2*log2(e) factor, we compute 2^s_x directly
        exp_term = self.eu(-s_x, use_log2e_scaling=False)  # Note: negative because s_x is negative
        
        # Stage 3: Division xi / (1 + 2^(-2*log2e*h(xi)))
        # The DU handles the "1 +" part when add_one_to_denominator=True
        result = self.du(x, exp_term, add_one_to_denominator=True)
        
        return result

# ==============================================================================
# Test Functions
# ==============================================================================

def test_scu_vs_standard_softmax():
    """Test SCU against standard PyTorch softmax"""
    print("Testing SCU vs Standard Softmax")
    print("-" * 40)
    
    batch_size, num_heads, seq_len = 2, 8, 16
    scores = torch.randn(batch_size, num_heads, seq_len, seq_len) * 2.0
    
    # Initialize both implementations
    scu = SCU(use_mask=False)
    
    with torch.no_grad():
        scu_output = scu(scores)
        standard_output = F.softmax(scores, dim=-1)
        
        # Compare outputs
        diff = torch.abs(scu_output - standard_output)
        max_diff = torch.max(diff)
        mean_diff = torch.mean(diff)
        
        print(f"Max difference: {max_diff:.6f}")
        print(f"Mean difference: {mean_diff:.6f}")
        
        # Check probability normalization
        scu_sums = torch.sum(scu_output, dim=-1)
        print(f"SCU probability sums (should be ~1.0): {scu_sums[0, 0, 0]:.6f}")
        print(f"Standard probability sums: {torch.sum(standard_output, dim=-1)[0, 0, 0]:.6f}")

def test_gcu_vs_standard_gelu():
    """Test GCU against standard PyTorch GELU"""
    print("\nTesting GCU vs Standard GELU")
    print("-" * 40)
    
    x = torch.randn(32, 128, 256) * 2.0  # Typical transformer hidden states
    
    # Initialize both implementations
    gcu = GCU()
    
    with torch.no_grad():
        gcu_output = gcu(x)
        standard_output = F.gelu(x)
        
        # Compare outputs
        diff = torch.abs(gcu_output - standard_output)
        max_diff = torch.max(diff)
        mean_diff = torch.mean(diff)
        
        print(f"Max difference: {max_diff:.6f}")
        print(f"Mean difference: {mean_diff:.6f}")
        
        # Check output ranges
        print(f"GCU output range: [{torch.min(gcu_output):.4f}, {torch.max(gcu_output):.4f}]")
        print(f"Standard GELU range: [{torch.min(standard_output):.4f}, {torch.max(standard_output):.4f}]")

def test_shared_components():
    """Test that EU and DU work correctly when shared"""
    print("\nTesting Shared EU and DU Components")
    print("-" * 40)
    
    # Test EU
    eu = ExponentialUnit()
    x = torch.randn(10) * 2.0
    
    eu_with_scaling = eu(x, use_log2e_scaling=True)
    eu_without_scaling = eu(x, use_log2e_scaling=False)
    standard_exp = torch.exp(x)
    standard_2pow = 2.0 ** x
    
    print(f"EU with log2e scaling vs exp: {torch.mean(torch.abs(eu_with_scaling - standard_exp)):.6f}")
    print(f"EU without scaling vs 2^x: {torch.mean(torch.abs(eu_without_scaling - standard_2pow)):.6f}")
    
    # Test DU
    du = DivisionUnit()
    numerator = torch.randn(10) + 1.0  # Avoid division by zero
    denominator = torch.randn(10) + 1.0
    
    du_output = du(numerator, denominator, add_one_to_denominator=False)
    standard_div = numerator / denominator
    
    print(f"DU vs standard division: {torch.mean(torch.abs(du_output - standard_div)):.6f}")

if __name__ == "__main__":
    test_scu_vs_standard_softmax()
    test_gcu_vs_standard_gelu() 
    test_shared_components()