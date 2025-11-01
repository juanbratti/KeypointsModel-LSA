"""
Complete Sign Language Translator Model

This module integrates all components to create the complete sign language
translation model as described in the paper:

Keypoints → Pose Encoder → Positional Encoding → nn.Transformer → Generator
                                                        ↓
Text ← Generator ← nn.Transformer ← Positional Encoding ← Text Embeddings

Updated to use PyTorch's native nn.Transformer for better performance and maintainability.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, Dict, Any
import math

from .pose_encoder import PoseEncoder
from .positional_encoding import PositionalEncoding

# Import Vocabulary class
try:
    from utils.vocabulary import Vocabulary
except ImportError:
    # Handle relative import for when called from different contexts
    import sys
    import os
    sys.path.append(os.path.dirname(os.path.dirname(__file__)))
    from utils.vocabulary import Vocabulary


class Generator(nn.Module):
    """
    Generator for converting transformer output to vocabulary probabilities
    
    Linear layer followed by log softmax to generate probability distribution
    over the vocabulary.
    """
    
    def __init__(self, d_model: int, vocab_size: int):
        """
        Initialize generator
        
        Args:
            d_model: Model dimension
            vocab_size: Size of vocabulary
        """
        super(Generator, self).__init__()
        
        self.linear = nn.Linear(d_model, vocab_size)
        
        # Initialize weights
        nn.init.xavier_uniform_(self.linear.weight)
        nn.init.zeros_(self.linear.bias)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Generate vocabulary probabilities
        
        Args:
            x: Transformer output (batch_size, seq_len, d_model)
            
        Returns:
            Vocabulary probabilities (batch_size, seq_len, vocab_size)
        """
        # Linear projection to vocabulary size
        logits = self.linear(x)
        
        # Apply log softmax for numerical stability during training
        return F.log_softmax(logits, dim=-1)


class SignLanguageTranslator(nn.Module):
    """
    Complete Sign Language Translation Model using PyTorch's nn.Transformer
    
    Architecture:
    1. Keypoints (T, 1086) → Pose Encoder → (T, d_model)
    2. Add Positional Encoding
    3. nn.Transformer → Contextual representations and text generation
    4. Generator → Vocabulary probabilities
    """
    
    def __init__(self,
                 vocab_size: int,
                 d_model: int = 256, # dal bianco paper.
                 encoder_layers: int = 2, # dal bianco paper
                 decoder_layers: int = 6, # dal bianco paper
                 num_heads: int = 8, # attention paper
                 d_ff: int = None,
                 dropout: float = 0.2, # dal bianco paper
                 max_seq_length: int = 5005, # max frame number (longest video is 5003 frames long)
                 keypoint_dim: int = 1086,  # 543 keypoints × 2 coordinates
                 pad_idx: int = 0): # padding token index.
        """
        Initialize Sign Language Translator using nn.Transformer
        
        Args:
            vocab_size: Size of vocabulary
            d_model: Model dimension
            encoder_layers: Number of encoder layers (1-2 based on dataset)
            decoder_layers: Number of decoder layers (2-6 based on dataset)
            num_heads: Number of attention heads
            d_ff: Feed-forward dimension (defaults to 4 * d_model)
            dropout: Dropout probability
            max_seq_length: Maximum sequence length
            keypoint_dim: Keypoint feature dimension (1086 = 543 × 2)
            pad_idx: Padding token index
        """
        super(SignLanguageTranslator, self).__init__()
        
        if d_ff is None:
            d_ff = 4 * d_model # standard from the attention paper
        
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.pad_idx = pad_idx
        
        # 1. Pose Encoder (3 Conv1D layers, kernel=1)
        self.pose_encoder = PoseEncoder(
            input_dim=keypoint_dim,
            d_model=d_model,
            dropout=dropout
        )
        
        # 2. Positional Encoding (for both pose sequences and text)
        self.pos_encoding = PositionalEncoding(
            d_model=d_model,
            max_length=max_seq_length,
            dropout=dropout
        )
        
        # 3. Text Embeddings
        self.text_embeddings = nn.Embedding(vocab_size, d_model, padding_idx=pad_idx)
        
        # 4. PyTorch Transformer (replaces custom encoder and decoder)
        self.transformer = nn.Transformer(
            d_model=d_model,
            nhead=num_heads,
            num_encoder_layers=encoder_layers,
            num_decoder_layers=decoder_layers,
            dim_feedforward=d_ff,
            dropout=dropout,
            batch_first=True  # Use batch_first=True for easier handling
        )
        
        # 5. Generator (Linear + LogSoftmax)
        self.generator = Generator(d_model, vocab_size)
        
        # Initialize embeddings
        self._init_embeddings()
    
    def _init_embeddings(self):
        """Initialize text embeddings"""
        nn.init.normal_(self.text_embeddings.weight, mean=0, std=self.d_model ** -0.5)
        # Set padding embedding to zero
        with torch.no_grad():
            self.text_embeddings.weight[self.pad_idx].fill_(0)
    
    def create_src_key_padding_mask(self, src: torch.Tensor, pad_value: float = 0.0) -> torch.Tensor:
        """
        Create source key padding mask for nn.Transformer
        
        Args:
            src: Source sequence (batch_size, src_len, d_model)
            pad_value: Value used for padding
            
        Returns:
            Source key padding mask (batch_size, src_len) - True for padding positions
        """
        # Check if all features in a position are pad_value (for keypoint sequences)
        mask = (src.abs().sum(dim=-1) == pad_value)  # True for padding positions
        return mask
    
    def create_tgt_key_padding_mask(self, tgt: torch.Tensor) -> torch.Tensor:
        """
        Create target key padding mask for nn.Transformer
        
        Args:
            tgt: Target sequence (batch_size, tgt_len)
            
        Returns:
            Target key padding mask (batch_size, tgt_len) - True for padding positions
        """
        return (tgt == self.pad_idx)  # True for padding positions
    
    def create_tgt_mask(self, tgt_len: int, device: torch.device) -> torch.Tensor:
        """
        Create target mask (causal mask) for nn.Transformer
        
        Args:
            tgt_len: Target sequence length
            device: Device to create mask on
            
        Returns:
            Target mask (tgt_len, tgt_len) - True for positions that should be masked
        """
        # Create upper triangular mask (True for positions that should be masked)
        mask = torch.triu(torch.ones(tgt_len, tgt_len, device=device), diagonal=1).bool()
        return mask
    
    def encode_poses(self, 
                    keypoints: torch.Tensor, 
                    src_key_padding_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Encode pose sequences using pose encoder and positional encoding
        
        Args:
            keypoints: Keypoint sequences (batch_size, seq_len, keypoint_dim)
            src_key_padding_mask: Source key padding mask (optional)
            
        Returns:
            Encoded pose representations (batch_size, seq_len, d_model)
        """
        
        # 1. Pose encoding (Conv1D layers)
        pose_features = self.pose_encoder(keypoints)  # (batch, seq_len, d_model)
        
        # 2. Add positional encoding
        pose_features = self.pos_encoding(pose_features)
        
        return pose_features
    
    def decode_text(self,
                   tgt_tokens: torch.Tensor,
                   encoder_output: torch.Tensor,
                   tgt_mask: Optional[torch.Tensor] = None,
                   tgt_key_padding_mask: Optional[torch.Tensor] = None,
                   memory_key_padding_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Decode text sequences using text embeddings and positional encoding
        
        Args:
            tgt_tokens: Target token indices (batch_size, tgt_len)
            encoder_output: Encoded pose representations (batch_size, src_len, d_model)
            tgt_mask: Target mask (causal mask)
            tgt_key_padding_mask: Target key padding mask
            memory_key_padding_mask: Memory key padding mask
            
        Returns:
            Target embeddings ready for transformer (batch_size, tgt_len, d_model)
        """
        # 1. Text embeddings
        tgt_embeddings = self.text_embeddings(tgt_tokens)  # (batch, tgt_len, d_model)
        
        # Scale embeddings (as in original Transformer paper)
        tgt_embeddings = tgt_embeddings * math.sqrt(self.d_model)
        
        # 2. Add positional encoding
        tgt_embeddings = self.pos_encoding(tgt_embeddings)
        
        return tgt_embeddings
    
    def forward(self,
                keypoints: torch.Tensor,
                tgt_tokens: torch.Tensor,
                attention_mask: Optional[torch.Tensor] = None,
                src_key_padding_mask: Optional[torch.Tensor] = None,
                tgt_key_padding_mask: Optional[torch.Tensor] = None,
                memory_key_padding_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Forward pass through the complete model using nn.Transformer
        
        Args:
            keypoints: Keypoint sequences (batch_size, src_len, keypoint_dim)
            tgt_tokens: Target token indices (batch_size, tgt_len)
            attention_mask: Attention mask for variable length keypoint sequences (batch_size, src_len) - DEPRECATED, use src_key_padding_mask
            src_key_padding_mask: Source key padding mask (batch_size, src_len) - True for padding positions
            tgt_key_padding_mask: Target key padding mask (batch_size, tgt_len) - True for padding positions
            memory_key_padding_mask: Memory key padding mask (batch_size, src_len) - True for padding positions
            
        Returns:
            Vocabulary probabilities (batch_size, tgt_len, vocab_size)
        """
        device = keypoints.device
        tgt_len = tgt_tokens.size(1)
        
        # 1. Encode poses (pose encoder + positional encoding)
        src = self.encode_poses(keypoints)  # (batch_size, src_len, d_model)
        
        # 2. Decode text (text embeddings + positional encoding)
        tgt = self.decode_text(tgt_tokens, src)  # (batch_size, tgt_len, d_model)
        
        # 3. Create masks for nn.Transformer
        # Create source key padding mask from keypoints if not provided
        if src_key_padding_mask is None:
            if attention_mask is not None:
                # Convert attention_mask (1 for real, 0 for padding) to padding mask (True for padding)
                src_key_padding_mask = (attention_mask == 0)
            else:
                src_key_padding_mask = self.create_src_key_padding_mask(src)
        
        # Create target key padding mask if not provided
        if tgt_key_padding_mask is None:
            tgt_key_padding_mask = self.create_tgt_key_padding_mask(tgt_tokens)
        
        # Use src_key_padding_mask as memory_key_padding_mask if not provided
        if memory_key_padding_mask is None:
            memory_key_padding_mask = src_key_padding_mask
        
        # Create target mask (causal mask)
        tgt_mask = self.create_tgt_mask(tgt_len, device)
        
        # 4. Pass through nn.Transformer
        transformer_output = self.transformer(
            src=src,
            tgt=tgt,
            tgt_mask=tgt_mask,
            src_key_padding_mask=src_key_padding_mask,
            tgt_key_padding_mask=tgt_key_padding_mask,
            memory_key_padding_mask=memory_key_padding_mask
        )
        
        # 5. Generate vocabulary probabilities
        vocab_probs = self.generator(transformer_output)
        
        return vocab_probs
    
    def generate_greedy(self,
                       keypoints: torch.Tensor,
                       vocabulary: Vocabulary,
                       max_length: int = 50,
                       src_key_padding_mask: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Generate text using greedy decoding with nn.Transformer
        
        Args:
            keypoints: Keypoint sequences (batch_size, src_len, keypoint_dim)
            vocabulary: Vocabulary for token conversion
            max_length: Maximum generation length
            src_key_padding_mask: Source key padding mask (optional)
            
        Returns:
            Tuple of (generated_tokens, log_probabilities)
        """
        self.eval()
        batch_size = keypoints.size(0)
        device = keypoints.device
        
        # Encode poses once
        with torch.no_grad():
            src = self.encode_poses(keypoints)  # (batch_size, src_len, d_model)
        
        # Create source key padding mask if not provided
        if src_key_padding_mask is None:
            src_key_padding_mask = self.create_src_key_padding_mask(src)
        
        # Initialize with SOS token
        generated_tokens = torch.full((batch_size, 1), vocabulary.SOS_IDX, 
                                    dtype=torch.long, device=device)
        log_probs = torch.zeros(batch_size, device=device)
        
        for step in range(max_length - 1):
            current_length = generated_tokens.size(1)
            
            # Prepare target embeddings
            tgt = self.decode_text(generated_tokens, src)  # (batch_size, current_length, d_model)
            
            # Create masks
            tgt_mask = self.create_tgt_mask(current_length, device)
            tgt_key_padding_mask = self.create_tgt_key_padding_mask(generated_tokens)
            
            # Pass through transformer
            with torch.no_grad():
                transformer_output = self.transformer(
                    src=src,
                    tgt=tgt,
                    tgt_mask=tgt_mask,
                    src_key_padding_mask=src_key_padding_mask,
                    tgt_key_padding_mask=tgt_key_padding_mask,
                    memory_key_padding_mask=src_key_padding_mask
                )
                
                # Generate vocabulary probabilities
                vocab_probs = self.generator(transformer_output)
            
            # Get next token probabilities
            next_token_probs = vocab_probs[:, -1, :]  # (batch_size, vocab_size)
            
            # Greedy selection
            next_tokens = torch.argmax(next_token_probs, dim=-1, keepdim=True)
            next_log_probs = torch.gather(next_token_probs, 1, next_tokens).squeeze(1)
            
            # Append to generated sequence
            generated_tokens = torch.cat([generated_tokens, next_tokens], dim=1)
            log_probs += next_log_probs
            
            # Check for EOS token
            if (next_tokens.squeeze(1) == vocabulary.EOS_IDX).all():
                break
        
        return generated_tokens, log_probs
    
    def generate_beam_search(self,
                           keypoints: torch.Tensor,
                           vocabulary: Vocabulary,
                           beam_size: int = 32,
                           max_length: int = 50,
                           src_key_padding_mask: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Generate text using beam search with nn.Transformer
        
        Args:
            keypoints: Keypoint sequences (1, src_len, keypoint_dim) - single sample
            vocabulary: Vocabulary for token conversion
            beam_size: Beam size for search
            max_length: Maximum generation length
            src_key_padding_mask: Source key padding mask (optional)
            
        Returns:
            Tuple of (best_sequence, best_score)
        """
        self.eval()
        device = keypoints.device
        
        if keypoints.size(0) != 1:
            raise ValueError("Beam search only supports batch_size=1")
        
        # Encode poses once
        with torch.no_grad():
            src = self.encode_poses(keypoints)  # (1, src_len, d_model)
        
        # Create source key padding mask if not provided
        if src_key_padding_mask is None:
            src_key_padding_mask = self.create_src_key_padding_mask(src)
        
        # Initialize beam
        # Each beam item: (sequence, log_prob)
        beam = [(torch.tensor([vocabulary.SOS_IDX], device=device), 0.0)]
        completed_sequences = []
        
        for step in range(max_length - 1):
            candidates = []
            
            for seq, score in beam:
                if seq[-1] == vocabulary.EOS_IDX:
                    completed_sequences.append((seq, score))
                    continue
                
                # Expand sequence
                seq_tensor = seq.unsqueeze(0)  # (1, seq_len)
                current_length = seq_tensor.size(1)
                
                # Prepare target embeddings
                tgt = self.decode_text(seq_tensor, src)  # (1, current_length, d_model)
                
                # Create masks
                tgt_mask = self.create_tgt_mask(current_length, device)
                tgt_key_padding_mask = self.create_tgt_key_padding_mask(seq_tensor)
                
                # Pass through transformer
                with torch.no_grad():
                    transformer_output = self.transformer(
                        src=src,
                        tgt=tgt,
                        tgt_mask=tgt_mask,
                        src_key_padding_mask=src_key_padding_mask,
                        tgt_key_padding_mask=tgt_key_padding_mask,
                        memory_key_padding_mask=src_key_padding_mask
                    )
                    
                    # Generate vocabulary probabilities
                    vocab_probs = self.generator(transformer_output)
                
                # Get probabilities for next token
                next_token_probs = vocab_probs[0, -1, :]  # (vocab_size,)
                
                # Get top-k candidates
                top_k_probs, top_k_indices = torch.topk(next_token_probs, beam_size)
                
                for prob, token_idx in zip(top_k_probs, top_k_indices):
                    new_seq = torch.cat([seq, token_idx.unsqueeze(0)])
                    new_score = score + prob.item()
                    candidates.append((new_seq, new_score))
            
            # Select top beam_size candidates
            candidates.sort(key=lambda x: x[1], reverse=True)
            beam = candidates[:beam_size]
            
            # Early stopping if all beams are completed
            if len(beam) == 0:
                break
        
        # Add remaining beams to completed sequences
        completed_sequences.extend(beam)
        
        # Return best sequence
        if completed_sequences:
            best_seq, best_score = max(completed_sequences, key=lambda x: x[1])
            return best_seq.unsqueeze(0), torch.tensor([best_score])
        else:
            # Fallback to greedy if no completed sequences
            return self.generate_greedy(keypoints, vocabulary, max_length, src_key_padding_mask)


def create_model_from_config(config: Dict[str, Any], vocab_size: int) -> SignLanguageTranslator:
    """
    Create model from configuration dictionary
    
    Args:
        config: Configuration dictionary
        vocab_size: Vocabulary size
        
    Returns:
        Configured SignLanguageTranslator model
    """
    return SignLanguageTranslator(
        vocab_size=vocab_size,
        d_model=config.get('d_model', 256),
        encoder_layers=config.get('encoder_layers', 2),
        decoder_layers=config.get('decoder_layers', 6),
        num_heads=config.get('num_heads', 8),
        d_ff=config.get('d_ff', None),
        dropout=config.get('dropout', 0.2),
        max_seq_length=config.get('max_seq_length', 5003),
        keypoint_dim=config.get('keypoint_dim', 1086),
        pad_idx=config.get('pad_idx', 0)
    )
