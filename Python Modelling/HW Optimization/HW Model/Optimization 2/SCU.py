import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import numpy as np

class FindMaxUnit(nn.Module):
    """
    Find Max Unit (FMU) - Hardware-friendly maximum finding using parallel comparison
    Based on the paper's Figure 7 design with O(log2 n) complexity
    """
    def __init__(self):
        super().__init__()
    
    def forward(self, x):
        """
        Find maximum value along the last dimension using parallel reduction
        Simulates the hardware tree-based comparison from the paper
        """
        # The paper mentions grouping elements for parallel processing
        # We simulate this with torch operations that can be hardware-optimized
        max_vals, _ = torch.max(x, dim=-1, keepdim=True)
        return max_vals

class ExponentialUnit(nn.Module):
    """
    Exponential Unit (EU) - Hardware-friendly base-2 exponential computation
    Based on the paper's piecewise linear approximation method
    """
    def __init__(self, num_segments=8):
        super().__init__()
        self.num_segments = num_segments
        
        # Pre-compute piecewise linear coefficients for 2^x approximation
        # Based on paper's approach: 2^frac(x) using lookup table
        self.register_buffer('segment_boundaries', torch.linspace(0, 1, num_segments + 1))
        
        # Coefficients for linear approximation in each segment
        # Ki and Bi values from the paper's LUT approach
        coeffs = []
        for i in range(num_segments):
            x_start = i / num_segments
            x_end = (i + 1) / num_segments
            x_mid = (x_start + x_end) / 2
            
            # Linear approximation: y = K*x + B
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
            use_log2e_scaling: If True, applies log2(e) scaling for e^x approximation
        """
        if use_log2e_scaling:
            # Convert e^x to 2^(log2(e)*x) as mentioned in paper
            log2e = math.log2(math.e)  # ≈ 1.4427
            x = x * log2e
        
        # Split into integer and fractional parts
        x_int = torch.floor(x).long()
        x_frac = x - x_int.float()
        
        # Clamp fractional part to [0, 1) for lookup table
        x_frac = torch.clamp(x_frac, 0, 0.999999)
        
        # Find which segment each fractional part belongs to
        segment_idx = torch.floor(x_frac * self.num_segments).long()
        segment_idx = torch.clamp(segment_idx, 0, self.num_segments - 1)
        
        # Get coefficients for each segment
        K = self.coefficients[segment_idx, 0]
        B = self.coefficients[segment_idx, 1]
        
        # Apply piecewise linear approximation: 2^frac(x)
        frac_result = K * x_frac + B
        
        # Apply integer part as bit shifts: 2^int(x) * 2^frac(x)
        # Clamp integer part to prevent overflow
        x_int_clamped = torch.clamp(x_int, -10, 10)
        
        # Simulate bit shift operation: multiply by 2^int_part
        result = frac_result * (2.0 ** x_int_clamped.float())
        
        return result

class DivisionUnit(nn.Module):
    """
    Division Unit (DU) - Hardware-friendly division using the paper's method
    Converts division to base-2 exponentiation as described in equations (11-12)
    """
    def __init__(self):
        super().__init__()
        self.eu = ExponentialUnit()
    
    def leading_one_detector(self, x):
        """
        Simulates Leading One Detector (LOD) to find w and m values
        Returns the position of the leading 1 bit
        """
        # For positive numbers, find the position of MSB
        # This simulates the hardware LOD function
        abs_x = torch.abs(x)
        
        # Avoid log of zero
        abs_x = torch.clamp(abs_x, min=1e-8)
        
        # Find integer part (w) and fractional part (m-1) 
        log2_x = torch.log2(abs_x)
        w = torch.floor(log2_x).long()
        m_minus_1 = log2_x - w.float()
        
        return w, m_minus_1 + 1.0  # m = (m-1) + 1

    def forward(self, numerator, denominator):
        """
        Perform division using the paper's method: F1/F2 = 2^((m1+w1)-(m2+w2))
        """
        # Get LOD values for numerator and denominator
        w1, m1 = self.leading_one_detector(numerator)
        w2, m2 = self.leading_one_detector(denominator)
        
        # Apply equation (12): F1/F2 = 2^((m1+w1)-(m2+w2))
        exponent = (m1 + w1.float()) - (m2 + w2.float())
        
        # Use exponential unit to compute 2^exponent
        result = self.eu(exponent, use_log2e_scaling=False)
        
        # Handle signs
        sign = torch.sign(numerator) * torch.sign(denominator)
        result = result * sign
        
        return result

class AdderTree(nn.Module):
    """
    Adder Tree for parallel summation
    Simulates the hardware adder tree structure
    """
    def __init__(self):
        super().__init__()
    
    def forward(self, x):
        """
        Sum along the last dimension using tree-based reduction
        """
        return torch.sum(x, dim=-1, keepdim=True)

class SCU(nn.Module):
    """
    Softmax Compute Unit (SCU) - Complete hardware-friendly softmax implementation
    Based on Figure 6 from the paper with all four stages
    """
    def __init__(self, use_mask=False):
        super().__init__()
        self.use_mask = use_mask
        
        # Initialize all submodules
        self.fmu = FindMaxUnit()          # Stage 1: Find maximum
        self.eu = ExponentialUnit()       # Stage 2: Exponential computation
        self.adder_tree = AdderTree()     # Stage 3: Sum computation
        self.du = DivisionUnit()          # Stage 4: Division computation
    
    def forward(self, scores, attention_mask=None):
        """
        Complete SCU dataflow as described in the paper
        
        Args:
            scores: Input attention scores [batch, heads, seq, seq]
            attention_mask: Optional attention mask for masking
        
        Returns:
            Softmax probabilities
        """
        original_shape = scores.shape
        
        # Apply attention mask if provided (before finding max)
        if attention_mask is not None and self.use_mask:
            # Convert mask to additive form (large negative values)
            mask_value = -10000.0
            scores = scores + attention_mask
        
        # Stage 1: Find maximum value (FMU)
        # Paper: "FMU receives input data and identifies the maximum value xmax"
        x_max = self.fmu(scores)  # [batch, heads, seq, 1]
        
        # Stage 2: Subtract max and compute exponential (EU)
        # Paper: "input data is subtracted from the output of FMU to obtain xi - xmax"
        shifted_scores = scores - x_max  # Numerical stability
        
        # Paper: "which is then fed into the EU for exponential computation"
        exp_scores = self.eu(shifted_scores, use_log2e_scaling=True)
        
        # Stage 3: Sum exponentials (Adder Tree)
        # Paper: "The Adder Tree calculates the cumulative sum of the EU outputs"
        sum_exp = self.adder_tree(exp_scores)  # [batch, heads, seq, 1]
        
        # Stage 4: Division (DU)
        # Paper: "DU receives the cumulative sum... to compute the exponent of Formula (11)"
        # Then: "The EU processes the output from DU and computes the final result"
        
        # Use our custom division unit for hardware-friendly division
        probabilities = self.du(exp_scores, sum_exp)
        
        # Ensure numerical stability and proper probability distribution
        probabilities = torch.clamp(probabilities, min=1e-8, max=1.0)
        
        # Renormalize to ensure sum to 1 (optional verification step)
        prob_sum = torch.sum(probabilities, dim=-1, keepdim=True)
        probabilities = probabilities / (prob_sum + 1e-8)
        
        return probabilities

# Comparison class to show the difference
class StandardSoftmax(nn.Module):
    """Standard PyTorch softmax for comparison"""
    def __init__(self):
        super().__init__()
    
    def forward(self, scores, attention_mask=None):
        if attention_mask is not None:
            scores = scores + attention_mask
        return F.softmax(scores, dim=-1)

# Test function to verify SCU implementation
def test_scu_vs_standard():
    """Test SCU against standard softmax"""
    batch_size, num_heads, seq_len = 2, 8, 16
    
    # Create test data
    scores = torch.randn(batch_size, num_heads, seq_len, seq_len) * 2.0
    
    # Initialize both softmax implementations
    scu = SCU()
    standard = StandardSoftmax()
    
    # Test without mask
    print("Testing SCU vs Standard Softmax (no mask):")
    with torch.no_grad():
        scu_output = scu(scores)
        standard_output = standard(scores)
        
        # Check if outputs are similar
        diff = torch.abs(scu_output - standard_output)
        max_diff = torch.max(diff)
        mean_diff = torch.mean(diff)
        
        print(f"Max difference: {max_diff:.6f}")
        print(f"Mean difference: {mean_diff:.6f}")
        
        # Check if probabilities sum to 1
        scu_sums = torch.sum(scu_output, dim=-1)
        standard_sums = torch.sum(standard_output, dim=-1)
        
        print(f"SCU probability sums (should be ~1.0): {scu_sums[0, 0, 0]:.6f}")
        print(f"Standard probability sums: {standard_sums[0, 0, 0]:.6f}")
    
    # Test with attention mask
    print("\nTesting with attention mask:")
    attention_mask = torch.zeros(batch_size, 1, 1, seq_len)
    attention_mask[:, :, :, seq_len//2:] = -10000.0  # Mask second half
    
    scu_masked = SCU(use_mask=True)
    
    with torch.no_grad():
        scu_output_masked = scu_masked(scores, attention_mask)
        standard_output_masked = standard(scores, attention_mask)
        
        diff_masked = torch.abs(scu_output_masked - standard_output_masked)
        max_diff_masked = torch.max(diff_masked)
        mean_diff_masked = torch.mean(diff_masked)
        
        print(f"Max difference (with mask): {max_diff_masked:.6f}")
        print(f"Mean difference (with mask): {mean_diff_masked:.6f}")

if __name__ == "__main__":
    test_scu_vs_standard()