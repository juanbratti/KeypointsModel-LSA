import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple


class PoseEncoder(nn.Module):
    def __init__(self, 
                 input_dim: int = 1086,  # 543 keypoints × 2 coordenadas
                 d_model: int = 256,     # Output dimension (paper)
                 dropout: float = 0.2):  # Dropout probability (paper)

        super(PoseEncoder, self).__init__()
        
        self.input_dim = input_dim
        self.d_model = d_model
        
        # Calculate intermediate dimensions
        # We'll use a gradual reduction: input_dim → d_model*4 → d_model*2 → d_model
        intermediate_dim1 = max(d_model * 4, d_model)
        intermediate_dim2 = max(d_model * 2, d_model)
        

        self.conv1 = nn.Conv1d(
            in_channels=input_dim,
            out_channels=intermediate_dim1,
            kernel_size=1, 
            bias=True
        )
        
        self.conv2 = nn.Conv1d(
            in_channels=intermediate_dim1,
            out_channels=intermediate_dim2,
            kernel_size=1,
            bias=True
        )
        
        self.conv3 = nn.Conv1d(
            in_channels=intermediate_dim2,
            out_channels=d_model,
            kernel_size=1,
            bias=True
        )
        
        # Dropout for regularization
        self.dropout = nn.Dropout(dropout)
        
        # Initialize weights
        self._init_weights()
    
    def _init_weights(self):
        """Initialize weights using Xavier initialization"""
        for conv in [self.conv1, self.conv2, self.conv3]:
            nn.init.xavier_uniform_(conv.weight)
            nn.init.zeros_(conv.bias)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the pose encoder
        
        Args:
            x: Input tensor of shape (batch_size, sequence_length, input_dim)
               where input_dim = 543 × 2 = 1086
            
        Returns:
            Encoded poses of shape (batch_size, sequence_length, d_model)
        """
        # Conv1D expects (batch_size, channels, sequence_length)
        # Transpose from (batch, seq, features) to (batch, features, seq)
        x = x.transpose(1, 2)  # (batch_size, input_dim, seq_len)
        
        # First Conv1D layer with temporal context, (batch_size, intermediate_dim1, seq_len)
        x = self.conv1(x) # transformamos de 1086 a 1024 con contexto temporal
        x = F.relu(x) 
        x = self.dropout(x) # regularización del 20% de neuronaes
        
        # Second Conv1D layer with temporal context, (batch_size, intermediate_dim2, seq_len)
        x = self.conv2(x) # transformamos de 1024 a 512 con contexto temporal
        x = F.relu(x)
        x = self.dropout(x)
        
        # Third Conv1D layer with temporal context, (batch_size, d_model, seq_len)
        x = self.conv3(x) # transformamos de 512 a 256 con contexto temporal
        x = F.relu(x)
        x = self.dropout(x)
        
        # Transpose back to (batch, seq, features)
        x = x.transpose(1, 2)  # (batch_size, seq_len, d_model)
        return x