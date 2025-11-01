"""
Inference and Text Generation for Sign Language Translation

This module provides inference utilities including greedy decoding and beam search
for generating text translations from sign language keypoint sequences.
"""

import torch
import torch.nn.functional as F
from typing import List, Tuple, Optional, Dict, Any
import numpy as np
from dataclasses import dataclass
import heapq

# Import necessary classes
try:
    from models.sign_language_translator import SignLanguageTranslator
    from utils.vocabulary import Vocabulary
except ImportError:
    # Handle relative imports
    import sys
    import os
    sys.path.append(os.path.dirname(os.path.dirname(__file__)))
    from models.sign_language_translator import SignLanguageTranslator
    from utils.vocabulary import Vocabulary

class SignLanguageGenerator:
    """
    Text generator for sign language translation
    
    Provides methods for generating text from keypoint sequences using
    different decoding strategies.
    """
    
    def __init__(self, 
                 model,
                 vocabulary,
                 device: torch.device = None):
        """
        Initialize generator
        
        Args:
            model: Trained SignLanguageTranslator model
            vocabulary: Vocabulary for token conversion
            device: Device for inference
        """
        self.model = model
        self.vocabulary = vocabulary
        self.device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Move model to device and set to eval mode
        self.model.to(self.device)
        self.model.eval()
    
    def preprocess_keypoints(self, keypoints: torch.Tensor) -> torch.Tensor:
        """
        Preprocess keypoints for inference
        
        Args:
            keypoints: Raw keypoint sequences
            
        Returns:
            Preprocessed keypoints ready for model
        """
        # Add batch dimension if needed
        if keypoints.dim() == 2:  # (seq_len, features)
            keypoints = keypoints.unsqueeze(0)
        
        # Move to device
        keypoints = keypoints.to(self.device)
        
        return keypoints
    
    def generate_greedy(self,
                       keypoints: torch.Tensor,
                       max_length: int = 50,
                       temperature: float = 1.0,
                       return_attention: bool = False) -> Dict[str, Any]:
        """
        Generate text using greedy decoding (delegating to model's built-in method)
        
        Args:
            keypoints: Keypoint sequences (batch_size, seq_len, keypoint_dim)
            max_length: Maximum generation length
            temperature: Sampling temperature (1.0 = no change, not used in greedy)
            return_attention: Whether to return attention weights (not implemented yet)
            
        Returns:
            Dictionary containing generated tokens, text, and optionally attention
        """
        keypoints = self.preprocess_keypoints(keypoints)
        
        # Use the model's built-in greedy generation
        with torch.no_grad():
            generated_tokens, log_probs = self.model.generate_greedy(
                keypoints, 
                self.vocabulary, 
                max_length=max_length
            )
        
        # Convert to text
        generated_texts = self.vocabulary.decode_batch(generated_tokens, remove_special_tokens=True)
        
        result = {
            'tokens': generated_tokens,
            'texts': generated_texts,
            'log_probs': log_probs,
            'scores': log_probs / generated_tokens.size(1)  # Length-normalized scores
        }
        
        return result
    
    def generate_beam_search(self,
                           keypoints: torch.Tensor,
                           beam_size: int = 5,
                           max_length: int = 50,
                           length_penalty: float = 1.0,
                           early_stopping: bool = True,
                           num_return_sequences: int = 1) -> Dict[str, Any]:
        """
        Generate text using beam search (delegating to model's built-in method)
        
        Args:
            keypoints: Keypoint sequences (1, seq_len, keypoint_dim) - single sample only
            beam_size: Beam size for search
            max_length: Maximum generation length
            length_penalty: Length penalty factor (not used in model's beam search yet)
            early_stopping: Whether to stop when EOS is generated (not used yet)
            num_return_sequences: Number of sequences to return (always 1 for now)
            
        Returns:
            Dictionary containing generated sequences and scores
        """
        keypoints = self.preprocess_keypoints(keypoints)
        
        if keypoints.size(0) != 1:
            raise ValueError("Beam search only supports batch_size=1")
        
        # Use the model's built-in beam search
        with torch.no_grad():
            best_tokens, best_score = self.model.generate_beam_search(
                keypoints, 
                self.vocabulary, 
                beam_size=beam_size,
                max_length=max_length
            )
        
        # Convert to text
        generated_text = self.vocabulary.decode_batch(best_tokens, remove_special_tokens=True)[0]
        
        result = {
            'tokens': [best_tokens],
            'texts': [generated_text],
            'scores': [best_score.item()],
            'num_sequences': 1
        }
        
        return result

def translate_sign_sequence(model,
                          vocabulary,
                          keypoints: torch.Tensor,
                          method: str = "beam_search",
                          **kwargs) -> Dict[str, Any]:
    """
    High-level function to translate sign language sequence to text
    
    Args:
        model: Trained SignLanguageTranslator model
        vocabulary: Vocabulary for token conversion
        keypoints: Keypoint sequences
        method: Generation method ("greedy", "beam_search", "nucleus")
        **kwargs: Additional arguments for generation method
        
    Returns:
        Translation results
    """
    generator = SignLanguageGenerator(model, vocabulary)
    
    if method == "greedy":
        return generator.generate_greedy(keypoints, **kwargs)
    elif method == "beam_search":
        return generator.generate_beam_search(keypoints, **kwargs)
    else:
        raise ValueError(f"Unknown generation method: {method}")
