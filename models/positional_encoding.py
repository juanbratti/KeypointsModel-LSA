"""
Positional Encoding for Sign Language Translation

This module implements sinusoidal positional encoding as used in the original
Transformer paper. It adds temporal information to both pose sequences and
text token embeddings.
"""

import torch
import torch.nn as nn
import math


class PositionalEncoding(nn.Module):
    """
    Sinusoidal Positional Encoding from "Attention Is All You Need"
    
    Adds positional information to input embeddings using sine and cosine
    functions of different frequencies.
    
    PE(pos, 2i) = sin(pos / 10000^(2i/d_model))
    PE(pos, 2i+1) = cos(pos / 10000^(2i/d_model))
    """
    
    def __init__(self, d_model: int, max_length: int, dropout: float):
        """
        Initialize positional encoding
        
        Args:
            d_model: Model dimension
            max_length: Maximum sequence length to precompute
            dropout: Dropout probability
        """
        super(PositionalEncoding, self).__init__()
        
        self.d_model = d_model
        self.dropout = nn.Dropout(dropout)
        
        # Create positional encoding matrix
        pe = torch.zeros(max_length, d_model)
        position = torch.arange(0, max_length, dtype=torch.float).unsqueeze(1)
        
        # Create division term for frequencies
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * 
                           (-math.log(10000.0) / d_model))
        
        # Apply sine to even indices
        pe[:, 0::2] = torch.sin(position * div_term)
        
        # Apply cosine to odd indices
        pe[:, 1::2] = torch.cos(position * div_term)
        
        # Add batch dimension and register as buffer
        pe = pe.unsqueeze(0).transpose(0, 1)  # (max_length, 1, d_model)
        self.register_buffer('pe', pe)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Add positional encoding to input embeddings
        
        Args:
            x: Input tensor of shape (batch_size, seq_len, d_model)
            
        Returns:
            Tensor with positional encoding added, same shape as input
        """
        # x shape: (batch_size, seq_len, d_model)
        seq_len = x.size(1)
        
        # Add positional encoding
        # pe[:seq_len, :] has shape (seq_len, 1, d_model)
        # Broadcasting adds it to all batch elements
        x = x + self.pe[:seq_len, :].transpose(0, 1)  # (1, seq_len, d_model)
        
        return self.dropout(x)

def create_positional_encoding(encoding_type: str = "sinusoidal",
                             d_model: int = 256,
                             max_length: int = 5003,
                             dropout: float = 0.1) -> nn.Module:
    """
    Factory function to create positional encoding
    
    Args:
        encoding_type: Type of encoding ("sinusoidal" or "learnable")
        d_model: Model dimension
        max_length: Maximum sequence length
        dropout: Dropout probability
        
    Returns:
        Positional encoding module
    """
    if encoding_type == "sinusoidal":
        return PositionalEncoding(d_model, max_length, dropout)
    raise ValueError(f"Unknown encoding type: {encoding_type}")
