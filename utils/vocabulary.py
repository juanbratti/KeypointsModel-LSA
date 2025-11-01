"""
Vocabulary and Tokenization Utilities for Sign Language Translation

This module provides utilities for building vocabularies, tokenizing text,
and converting between text and token indices for the sign language translation model.
"""

import torch
import pickle
import json
from typing import List, Dict, Set, Optional, Tuple
from collections import Counter, defaultdict
import re


class Vocabulary:
    """
    Vocabulary class for managing word-to-index and index-to-word mappings
    
    Handles special tokens like <PAD>, <SOS>, <EOS>, <UNK> and provides
    methods for encoding/decoding text sequences.
    """
    
    # Special tokens
    PAD_TOKEN = '<PAD>'
    SOS_TOKEN = '<SOS>'  # Start of sequence
    EOS_TOKEN = '<EOS>'  # End of sequence
    UNK_TOKEN = '<UNK>'  # Unknown token
    
    PAD_IDX = 0
    SOS_IDX = 1
    EOS_IDX = 2
    UNK_IDX = 3
    
    def __init__(self, 
                 min_freq: int = 1,
                 max_vocab_size: Optional[int] = None,
                 lowercase: bool = True):
        """
        Initialize vocabulary
        
        Args:
            min_freq: Minimum frequency for a word to be included
            max_vocab_size: Maximum vocabulary size (None for no limit)
            lowercase: Whether to convert text to lowercase
        """
        self.min_freq = min_freq
        self.max_vocab_size = max_vocab_size
        self.lowercase = lowercase
        
        # Initialize with special tokens
        self.word2idx = {
            self.PAD_TOKEN: self.PAD_IDX,
            self.SOS_TOKEN: self.SOS_IDX,
            self.EOS_TOKEN: self.EOS_IDX,
            self.UNK_TOKEN: self.UNK_IDX
        }
        
        self.idx2word = {
            self.PAD_IDX: self.PAD_TOKEN,
            self.SOS_IDX: self.SOS_TOKEN,
            self.EOS_IDX: self.EOS_TOKEN,
            self.UNK_IDX: self.UNK_TOKEN
        }
        
        self.word_freq = Counter()
        self._is_built = False
    
    def add_sentence(self, sentence: str) -> None:
        """
        Add a sentence to the vocabulary frequency counter
        
        Args:
            sentence: Input sentence to add
        """
        words = self.tokenize(sentence)
        self.word_freq.update(words)
    
    def add_sentences(self, sentences: List[str]) -> None:
        """
        Add multiple sentences to the vocabulary
        
        Args:
            sentences: List of sentences to add
        """
        for sentence in sentences:
            self.add_sentence(sentence)
    
    def build_vocab(self) -> None:
        """
        Build vocabulary from collected word frequencies
        """
        # Get words that meet minimum frequency requirement
        valid_words = [word for word, freq in self.word_freq.items() 
                      if freq >= self.min_freq]
        
        # Sort by frequency (descending)
        valid_words.sort(key=lambda x: self.word_freq[x], reverse=True)
        
        # Apply max vocabulary size limit
        if self.max_vocab_size is not None:
            # Reserve space for special tokens
            max_words = self.max_vocab_size - len(self.word2idx)
            valid_words = valid_words[:max_words]
        
        # Add words to vocabulary
        for word in valid_words:
            if word not in self.word2idx:
                idx = len(self.word2idx)
                self.word2idx[word] = idx
                self.idx2word[idx] = word
        
        self._is_built = True
        
        print(f"Built vocabulary with {len(self.word2idx)} words")
        print(f"Most frequent words: {valid_words[:10]}")
    
    def tokenize(self, text: str) -> List[str]:
        """
        Tokenize text into words
        
        Args:
            text: Input text to tokenize
            
        Returns:
            List of tokens
        """
        if self.lowercase:
            text = text.lower()
        
        # Simple tokenization - split on whitespace and punctuation
        # You might want to use more sophisticated tokenization
        tokens = re.findall(r'\b\w+\b|[^\w\s]', text)
        
        return tokens
    
    def encode(self, text: str, add_special_tokens: bool = True) -> List[int]:
        """
        Convert text to list of token indices
        
        Args:
            text: Input text to encode
            add_special_tokens: Whether to add SOS/EOS tokens
            
        Returns:
            List of token indices
        """
        if not self._is_built:
            raise ValueError("Vocabulary not built. Call build_vocab() first.")
        
        tokens = self.tokenize(text)
        
        # Convert tokens to indices
        indices = []
        
        if add_special_tokens:
            indices.append(self.SOS_IDX)
        
        for token in tokens:
            idx = self.word2idx.get(token, self.UNK_IDX)
            indices.append(idx)
        
        if add_special_tokens:
            indices.append(self.EOS_IDX)
        
        return indices
    
    def decode(self, indices: List[int], remove_special_tokens: bool = True) -> str:
        """
        Convert list of token indices back to text
        
        Args:
            indices: List of token indices
            remove_special_tokens: Whether to remove special tokens
            
        Returns:
            Decoded text string
        """
        words = []
        
        for idx in indices:
            if idx in self.idx2word:
                word = self.idx2word[idx]
                
                # Skip special tokens if requested
                if remove_special_tokens and word in [self.PAD_TOKEN, self.SOS_TOKEN, self.EOS_TOKEN]:
                    continue
                
                words.append(word)
            else:
                words.append(self.UNK_TOKEN)
        
        return ' '.join(words)
    
    def encode_batch(self, texts: List[str], 
                    max_length: Optional[int] = None,
                    add_special_tokens: bool = True,
                    padding: bool = True) -> torch.Tensor:
        """
        Encode a batch of texts to tensor
        
        Args:
            texts: List of text strings
            max_length: Maximum sequence length (None for auto)
            add_special_tokens: Whether to add SOS/EOS tokens
            padding: Whether to pad sequences to same length
            
        Returns:
            Tensor of token indices (batch_size, seq_len)
        """
        # Encode all texts
        encoded_texts = [self.encode(text, add_special_tokens) for text in texts]
        
        if max_length is None:
            max_length = max(len(seq) for seq in encoded_texts)
        
        if padding:
            # Pad sequences to same length
            padded_texts = []
            for seq in encoded_texts:
                if len(seq) > max_length:
                    # Truncate long sequences (keep SOS, truncate middle, keep EOS)
                    if add_special_tokens:
                        seq = [seq[0]] + seq[1:max_length-1] + [seq[-1]]
                    else:
                        seq = seq[:max_length]
                else:
                    # Pad short sequences
                    seq = seq + [self.PAD_IDX] * (max_length - len(seq))
                
                padded_texts.append(seq)
            
            return torch.tensor(padded_texts, dtype=torch.long)
        else:
            # Return list of tensors with different lengths
            return [torch.tensor(seq, dtype=torch.long) for seq in encoded_texts]
    
    def decode_batch(self, tensor: torch.Tensor, 
                    remove_special_tokens: bool = True) -> List[str]:
        """
        Decode a batch of token tensors to text
        
        Args:
            tensor: Tensor of token indices (batch_size, seq_len)
            remove_special_tokens: Whether to remove special tokens
            
        Returns:
            List of decoded text strings
        """
        if tensor.dim() == 1:
            tensor = tensor.unsqueeze(0)
        
        decoded_texts = []
        for seq in tensor:
            # Convert to list and decode
            indices = seq.tolist()
            text = self.decode(indices, remove_special_tokens)
            decoded_texts.append(text)
        
        return decoded_texts
    
    def __len__(self) -> int:
        """Return vocabulary size"""
        return len(self.word2idx)
    
    def save(self, filepath: str) -> None:
        """
        Save vocabulary to file
        
        Args:
            filepath: Path to save vocabulary
        """
        vocab_data = {
            'word2idx': self.word2idx,
            'idx2word': self.idx2word,
            'word_freq': dict(self.word_freq),
            'min_freq': self.min_freq,
            'max_vocab_size': self.max_vocab_size,
            'lowercase': self.lowercase,
            '_is_built': self._is_built
        }
        
        if filepath.endswith('.json'):
            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump(vocab_data, f, ensure_ascii=False, indent=2)
        else:
            with open(filepath, 'wb') as f:
                pickle.dump(vocab_data, f)
        
        print(f"Vocabulary saved to {filepath}")
    
    @classmethod
    def load(cls, filepath: str) -> 'Vocabulary':
        """
        Load vocabulary from file
        
        Args:
            filepath: Path to vocabulary file
            
        Returns:
            Loaded vocabulary instance
        """
        if filepath.endswith('.json'):
            with open(filepath, 'r', encoding='utf-8') as f:
                vocab_data = json.load(f)
        else:
            with open(filepath, 'rb') as f:
                vocab_data = pickle.load(f)
        
        # Create vocabulary instance
        vocab = cls(
            min_freq=vocab_data['min_freq'],
            max_vocab_size=vocab_data['max_vocab_size'],
            lowercase=vocab_data['lowercase']
        )
        
        # Restore state
        vocab.word2idx = vocab_data['word2idx']
        vocab.idx2word = {int(k): v for k, v in vocab_data['idx2word'].items()}
        vocab.word_freq = Counter(vocab_data['word_freq'])
        vocab._is_built = vocab_data['_is_built']
        
        print(f"Vocabulary loaded from {filepath} ({len(vocab)} words)")
        return vocab
    
    def get_word_frequencies(self, top_k: int = 50) -> List[Tuple[str, int]]:
        """
        Get most frequent words
        
        Args:
            top_k: Number of top words to return
            
        Returns:
            List of (word, frequency) tuples
        """
        return self.word_freq.most_common(top_k)
    
    def print_stats(self) -> None:
        """Print vocabulary statistics"""
        print("Vocabulary Statistics")
        print("=" * 40)
        print(f"Total words: {len(self.word2idx)}")
        print(f"Special tokens: {len([t for t in self.word2idx if t.startswith('<')])}")
        print(f"Regular words: {len(self.word2idx) - 4}")
        print(f"Min frequency: {self.min_freq}")
        print(f"Max vocab size: {self.max_vocab_size}")
        print(f"Lowercase: {self.lowercase}")
        print(f"Built: {self._is_built}")
        
        if self.word_freq:
            print(f"\nMost frequent words:")
            for word, freq in self.word_freq.most_common(10):
                print(f"  {word}: {freq}")


def build_vocabulary_from_texts(texts: List[str], 
                               min_freq: int = 1,
                               max_vocab_size: Optional[int] = None,
                               lowercase: bool = True) -> Vocabulary:
    """
    Build vocabulary from a list of texts
    
    Args:
        texts: List of text strings
        min_freq: Minimum frequency for inclusion
        max_vocab_size: Maximum vocabulary size
        lowercase: Whether to lowercase text
        
    Returns:
        Built vocabulary instance
    """
    vocab = Vocabulary(min_freq=min_freq, 
                      max_vocab_size=max_vocab_size,
                      lowercase=lowercase)
    
    print(f"Building vocabulary from {len(texts)} texts...")
    vocab.add_sentences(texts)
    vocab.build_vocab()
    
    return vocab


if __name__ == "__main__":
    # Test vocabulary functionality
    print("Testing Vocabulary")
    print("=" * 50)
    
    # Sample texts for testing
    sample_texts = [
        "Hello world, how are you?",
        "I am fine, thank you.",
        "This is a test sentence.",
        "Hello again, world!",
        "How are you doing today?",
        "Thank you very much.",
        "This is another test.",
    ]
    
    # Build vocabulary
    vocab = build_vocabulary_from_texts(
        sample_texts, 
        min_freq=1, 
        max_vocab_size=50,
        lowercase=True
    )
    
    # Print statistics
    vocab.print_stats()
    
    # Test encoding/decoding
    print("\nTesting encoding/decoding...")
    
    test_text = "Hello world, this is a test!"
    print(f"Original text: '{test_text}'")
    
    # Encode
    encoded = vocab.encode(test_text)
    print(f"Encoded: {encoded}")
    
    # Decode
    decoded = vocab.decode(encoded)
    print(f"Decoded: '{decoded}'")
    
    # Test batch processing
    print("\nTesting batch processing...")
    
    batch_texts = [
        "Hello world",
        "This is a longer sentence with more words",
        "Short text"
    ]
    
    # Encode batch
    batch_tensor = vocab.encode_batch(batch_texts, max_length=10, padding=True)
    print(f"Batch tensor shape: {batch_tensor.shape}")
    print(f"Batch tensor:\n{batch_tensor}")
    
    # Decode batch
    decoded_batch = vocab.decode_batch(batch_tensor)
    print("Decoded batch:")
    for i, text in enumerate(decoded_batch):
        print(f"  {i}: '{text}'")
    
    # Test saving/loading
    print("\nTesting save/load...")
    
    # Save vocabulary
    vocab.save("test_vocab.json")
    
    # Load vocabulary
    loaded_vocab = Vocabulary.load("test_vocab.json")
    
    # Test that loaded vocab works the same
    test_encoded = loaded_vocab.encode("Hello world")
    test_decoded = loaded_vocab.decode(test_encoded)
    print(f"Loaded vocab test: '{test_decoded}'")
    
    # Clean up
    import os
    if os.path.exists("test_vocab.json"):
        os.remove("test_vocab.json")
    
    print("\nAll vocabulary tests passed!")
