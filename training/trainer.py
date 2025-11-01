"""
Training utilities for Sign Language Translation

This module provides training loop implementation with teacher forcing,
loss computation, and model evaluation for the sign language translator.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.optim import Optimizer
from torch.optim.lr_scheduler import _LRScheduler
from typing import Dict, List, Optional, Tuple, Any
import time
import logging
from tqdm import tqdm
import numpy as np

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


class SignLanguageTrainer:
    """
    Trainer class for Sign Language Translation model
    
    Handles training with teacher forcing, validation, and model checkpointing.
    """
    
    def __init__(self,
                 model: SignLanguageTranslator,
                 vocabulary: Vocabulary,
                 optimizer: Optimizer,
                 criterion: nn.Module = None,
                 scheduler: Optional[_LRScheduler] = None,
                 device: torch.device = None,
                 grad_clip: float = 1.0,
                 log_interval: int = 100,
                 save_dir: str = "checkpoints"):
        """
        Initialize trainer
        
        Args:
            model: Sign language translation model
            vocabulary: Vocabulary for text processing
            optimizer: Optimizer for training
            criterion: Loss function (defaults to CrossEntropyLoss)
            scheduler: Learning rate scheduler (optional)
            device: Device for training
            grad_clip: Gradient clipping value
            log_interval: Logging interval in steps
            save_dir: Directory to save checkpoints
        """
        self.model = model
        self.vocabulary = vocabulary
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.grad_clip = grad_clip
        self.log_interval = log_interval
        self.save_dir = save_dir
        
        # Default criterion: CrossEntropyLoss with label smoothing
        if criterion is None:
            self.criterion = nn.CrossEntropyLoss(
                ignore_index=vocabulary.PAD_IDX,
                label_smoothing=0.1
            )
        else:
            self.criterion = criterion
        
        # Move model to device
        self.model.to(self.device)
        
        # Training statistics
        self.global_step = 0
        self.epoch = 0
        self.best_val_loss = float('inf')
        self.train_losses = []
        self.val_losses = []
        
        # Setup logging
        self.logger = logging.getLogger(__name__)
        
        # Create save directory
        import os
        os.makedirs(save_dir, exist_ok=True)
    
    def compute_loss(self, 
                    keypoints: torch.Tensor,
                    tgt_tokens: torch.Tensor,
                    attention_mask: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        Compute training loss with teacher forcing - UPDATED for variable length sequences
        
        Args:
            keypoints: Keypoint sequences (batch_size, src_len, keypoint_dim)
            tgt_tokens: Target token sequences (batch_size, tgt_len)
            attention_mask: Attention mask for variable length sequences (batch_size, src_len)
            
        Returns:
            Tuple of (loss, metrics_dict)
        """
        batch_size, tgt_len = tgt_tokens.shape
        
        # Prepare input and target for teacher forcing
        # Input: all tokens except the last one
        # Target: all tokens except the first one (SOS)
        tgt_input = tgt_tokens[:, :-1]  # Remove last token
        tgt_output = tgt_tokens[:, 1:]  # Remove first token (SOS)
        
        # Validate input data
        if torch.isnan(keypoints).any() or torch.isinf(keypoints).any():
            print(f"NaN/Inf in input keypoints!")
            print(f"Keypoints: min={keypoints.min().item():.6f}, max={keypoints.max().item():.6f}")
            raise ValueError("Invalid input keypoints!")
        
        # Create masks for nn.Transformer
        # Convert attention_mask to src_key_padding_mask format if provided
        src_key_padding_mask = None
        if attention_mask is not None:
            # attention_mask: 1 for real tokens, 0 for padding
            # src_key_padding_mask: True for padding tokens, False for real tokens
            src_key_padding_mask = (attention_mask == 0)
        
        # Create target key padding mask
        tgt_key_padding_mask = (tgt_input == self.vocabulary.PAD_IDX)
        
        # Forward pass with new mask interface
        vocab_probs = self.model(keypoints, tgt_input, 
                               src_key_padding_mask=src_key_padding_mask,
                               tgt_key_padding_mask=tgt_key_padding_mask,
                               memory_key_padding_mask=src_key_padding_mask)
        
        # Reshape for loss computation
        vocab_probs = vocab_probs.reshape(-1, vocab_probs.size(-1))  # (batch*seq, vocab)
        tgt_output = tgt_output.reshape(-1)  # (batch*seq,)
        
        # Compute loss
        loss = self.criterion(vocab_probs, tgt_output)
        
        # Check for NaN/Inf in critical tensors
        if torch.isnan(loss) or torch.isinf(loss):
            print(f"NaN/Inf detected at step {self.global_step}!")
            print(f"Loss: {loss.item()}")
            print(f"vocab_probs: min={vocab_probs.min().item():.6f}, max={vocab_probs.max().item():.6f}, mean={vocab_probs.mean().item():.6f}")
            print(f"vocab_probs has NaN: {torch.isnan(vocab_probs).any().item()}")
            print(f"tgt_output: min={tgt_output.min().item()}, max={tgt_output.max().item()}")
            
            # Check model parameters for NaN
            for name, param in self.model.named_parameters():
                if torch.isnan(param).any():
                    print(f"NaN in parameter {name}")
                if param.grad is not None and torch.isnan(param.grad).any():
                    print(f"NaN in gradient {name}")
            
            raise ValueError("NaN/Inf loss detected!")
        
        # Compute accuracy (excluding padding tokens)
        with torch.no_grad():
            pred_tokens = torch.argmax(vocab_probs, dim=-1)
            mask = (tgt_output != self.vocabulary.PAD_IDX)
            correct = (pred_tokens == tgt_output) & mask.bool()
            correct = correct.float()
            accuracy = correct.sum().float() / mask.sum().float()
        
        
        metrics = {
            'loss': loss.item(),
            'accuracy': accuracy.item(),
        }
        
        return loss, metrics
    
    def train_epoch(self, train_loader: DataLoader) -> Dict[str, float]:
        """
        Train for one epoch
        
        Args:
            train_loader: Training data loader
            
        Returns:
            Dictionary of epoch metrics
        """
        self.model.train()
        epoch_metrics = {'loss': 0.0, 'accuracy': 0.0}
        num_batches = len(train_loader)
        
        progress_bar = tqdm(train_loader, desc=f"Epoch {self.epoch}")
        
        for batch_idx, batch_data in enumerate(progress_bar):
            # Handle new batch format from variable_length_collate_fn
            if isinstance(batch_data, dict):
                # New format with attention masks
                keypoints = batch_data['keypoints'].to(self.device)
                attention_masks = batch_data['attention_masks'].to(self.device)
                texts = batch_data['transcriptions']
                sequence_lengths = batch_data['sequence_lengths']
            else:
                # Old format for compatibility
                keypoints, texts = batch_data
                keypoints = keypoints.to(self.device)
                attention_masks = None
                sequence_lengths = None
            
            # Encode text to tokens
            if isinstance(texts[0], str):
                tgt_tokens = self.vocabulary.encode_batch(
                    texts, max_length=None, add_special_tokens=True, padding=True
                ).to(self.device)
            else:
                tgt_tokens = texts.to(self.device)
            
            # Zero gradients
            self.optimizer.zero_grad()
            
            # Compute loss with attention mask
            loss, metrics = self.compute_loss(keypoints, tgt_tokens, attention_masks)
            
            # Backward pass
            loss.backward()
            
            # Gradient clipping
            if self.grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip)
            
            # Optimizer step
            self.optimizer.step()
            
            # Update metrics
            for key, value in metrics.items():
                epoch_metrics[key] += value
            
            # Update progress bar
            progress_bar.set_postfix({
                'loss': f"{metrics['loss']:.4f}",
                'acc': f"{metrics['accuracy']:.3f}",
            })
            
            # Logging
            if (batch_idx + 1) % self.log_interval == 0:
                self.logger.info(
                    f"Epoch {self.epoch}, Step {self.global_step}: "
                    f"Loss={metrics['loss']:.4f}, "
                    f"Acc={metrics['accuracy']:.3f}, "
                )
            
            self.global_step += 1
        
        # Average metrics over epoch
        for key in epoch_metrics:
            epoch_metrics[key] /= num_batches
        
        return epoch_metrics
    
    def validate(self, val_loader: DataLoader) -> Dict[str, float]:
        """
        Validate the model
        
        Args:
            val_loader: Validation data loader
            
        Returns:
            Dictionary of validation metrics
        """
        self.model.eval()
        val_metrics = {'loss': 0.0, 'accuracy': 0.0}
        num_batches = len(val_loader)
        
        with torch.no_grad():
            for batch_data in tqdm(val_loader, desc="Validation"):
                # Handle new batch format from variable_length_collate_fn
                if isinstance(batch_data, dict):
                    # New format with attention masks
                    keypoints = batch_data['keypoints'].to(self.device)
                    attention_masks = batch_data['attention_masks'].to(self.device)
                    texts = batch_data['transcriptions']
                    sequence_lengths = batch_data['sequence_lengths']
                else:
                    # Old format for compatibility
                    keypoints, texts = batch_data
                    keypoints = keypoints.to(self.device)
                    attention_masks = None
                    sequence_lengths = None
                
                # Encode text to tokens
                if isinstance(texts[0], str):
                    tgt_tokens = self.vocabulary.encode_batch(
                        texts, max_length=None, add_special_tokens=True, padding=True
                    ).to(self.device)
                else:
                    tgt_tokens = texts.to(self.device)
                
                # Compute loss with attention mask
                loss, metrics = self.compute_loss(keypoints, tgt_tokens, attention_masks)
                
                # Update metrics
                for key, value in metrics.items():
                    val_metrics[key] += value
        
        # Average metrics
        for key in val_metrics:
            val_metrics[key] /= num_batches
        
        return val_metrics
    
    def train(self,
              train_loader: DataLoader,
              val_loader: Optional[DataLoader] = None,
              num_epochs: int = 10,
              save_every: int = 1,
              early_stopping_patience: int = 5) -> Dict[str, List[float]]:
        """
        Complete training loop
        
        Args:
            train_loader: Training data loader
            val_loader: Validation data loader (optional)
            num_epochs: Number of epochs to train
            save_every: Save checkpoint every N epochs
            early_stopping_patience: Early stopping patience
            
        Returns:
            Training history dictionary
        """
        self.logger.info(f"Starting training for {num_epochs} epochs")
        self.logger.info(f"Model parameters: {sum(p.numel() for p in self.model.parameters()):,}")
        self.logger.info(f"Training on device: {self.device}")
        
        history = {'train_loss': [], 'train_acc': [], 'val_loss': [], 'val_acc': []}
        patience_counter = 0
        
        for epoch in range(num_epochs):
            self.epoch = epoch
            start_time = time.time()
            
            # Training
            train_metrics = self.train_epoch(train_loader)
            
            # Validation
            if val_loader is not None:
                val_metrics = self.validate(val_loader)
            else:
                val_metrics = {'loss': 0.0, 'accuracy': 0.0}
            
            # Update learning rate
            if self.scheduler is not None:
                if hasattr(self.scheduler, 'step'):
                    if 'ReduceLROnPlateau' in str(type(self.scheduler)):
                        self.scheduler.step(val_metrics['loss'])
                    else:
                        self.scheduler.step()
            
            # Update history
            history['train_loss'].append(train_metrics['loss'])
            history['train_acc'].append(train_metrics['accuracy'])
            history['val_loss'].append(val_metrics['loss'])
            history['val_acc'].append(val_metrics['accuracy'])
            
            # Logging
            epoch_time = time.time() - start_time
            self.logger.info(
                f"Epoch {epoch + 1}/{num_epochs} completed in {epoch_time:.2f}s - "
                f"Train Loss: {train_metrics['loss']:.4f}, "
                f"Train Acc: {train_metrics['accuracy']:.3f}, "
                f"Val Loss: {val_metrics['loss']:.4f}, "
                f"Val Acc: {val_metrics['accuracy']:.3f}"
            )
            
            # Save checkpoint
            if (epoch + 1) % save_every == 0:
                self.save_checkpoint(
                    epoch=epoch,
                    train_loss=train_metrics['loss'],
                    val_loss=val_metrics['loss']
                )
            
            # Early stopping
            if val_loader is not None:
                if val_metrics['loss'] < self.best_val_loss:
                    self.best_val_loss = val_metrics['loss']
                    patience_counter = 0
                    # Save best model
                    self.save_checkpoint(
                        epoch=epoch,
                        train_loss=train_metrics['loss'],
                        val_loss=val_metrics['loss'],
                        is_best=True
                    )
                else:
                    patience_counter += 1
                
                if patience_counter >= early_stopping_patience:
                    self.logger.info(f"Early stopping triggered after {patience_counter} epochs without improvement")
                    break
        
        self.logger.info("Training completed!")
        return history
    
    def save_checkpoint(self,
                       epoch: int,
                       train_loss: float,
                       val_loss: float,
                       is_best: bool = False):
        """
        Save model checkpoint
        
        Args:
            epoch: Current epoch
            train_loss: Training loss
            val_loss: Validation loss
            is_best: Whether this is the best model so far
        """
        checkpoint = {
            'epoch': epoch,
            'global_step': self.global_step,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'train_loss': train_loss,
            'val_loss': val_loss,
            'best_val_loss': self.best_val_loss,
            'vocab_size': self.model.vocab_size,
            'd_model': self.model.d_model
        }
        
        if self.scheduler is not None:
            checkpoint['scheduler_state_dict'] = self.scheduler.state_dict()
        
        # Save regular checkpoint
        checkpoint_path = f"{self.save_dir}/checkpoint_epoch_{epoch + 1}.pt"
        torch.save(checkpoint, checkpoint_path)
        
        # Save best model
        if is_best:
            best_path = f"{self.save_dir}/best_model.pt"
            torch.save(checkpoint, best_path)
            self.logger.info(f"New best model saved with validation loss: {val_loss:.4f}")
        
        self.logger.info(f"Checkpoint saved: {checkpoint_path}")
    
    def load_checkpoint(self, checkpoint_path: str, load_optimizer: bool = True):
        """
        Load model checkpoint
        
        Args:
            checkpoint_path: Path to checkpoint file
            load_optimizer: Whether to load optimizer state
        """
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        
        # Load model state
        self.model.load_state_dict(checkpoint['model_state_dict'])
        
        # Load optimizer state
        if load_optimizer and 'optimizer_state_dict' in checkpoint:
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        
        # Load scheduler state
        if self.scheduler is not None and 'scheduler_state_dict' in checkpoint:
            self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        
        # Restore training state
        self.epoch = checkpoint.get('epoch', 0)
        self.global_step = checkpoint.get('global_step', 0)
        self.best_val_loss = checkpoint.get('best_val_loss', float('inf'))
        
        self.logger.info(f"Checkpoint loaded from {checkpoint_path}")
        self.logger.info(f"Resuming from epoch {self.epoch + 1}, step {self.global_step}")


def create_trainer(model: SignLanguageTranslator,
                  vocabulary: Vocabulary,
                  learning_rate: float = 1e-4,
                  weight_decay: float = 0.01,
                  scheduler_type: str = "cosine",
                  **kwargs) -> SignLanguageTrainer:
    """
    Create trainer with default optimizer and scheduler
    
    Args:
        model: Sign language translation model
        vocabulary: Vocabulary for text processing
        learning_rate: Learning rate for optimizer
        weight_decay: Weight decay for optimizer
        scheduler_type: Type of scheduler ("cosine", "plateau", "none")
        **kwargs: Additional arguments for trainer
        
    Returns:
        Configured trainer instance
    """
    # Create optimizer
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=learning_rate,
        weight_decay=weight_decay,
        betas=(0.9, 0.98),
        eps=1e-9
    )
    
    # Create scheduler
    scheduler = None
    if scheduler_type == "cosine":
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=kwargs.get('num_epochs', 100)
        )
    elif scheduler_type == "plateau":
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', factor=0.5, patience=5, verbose=True
        )
    
    return SignLanguageTrainer(
        model=model,
        vocabulary=vocabulary,
        optimizer=optimizer,
        scheduler=scheduler,
        **kwargs
    )


if __name__ == "__main__":
    # Test trainer functionality
    print("Testing Sign Language Trainer")
    print("=" * 50)
    
    # This would normally be tested with actual data
    # For now, just verify the trainer can be created
    
    from ..models.sign_language_translator import SignLanguageTranslator
    from ..utils.vocabulary import Vocabulary
    
    # Create dummy model and vocabulary
    vocab_size = 1000
    model = SignLanguageTranslator(
        vocab_size=vocab_size,
        d_model=64,
        encoder_layers=1,
        decoder_layers=2,
        dropout=0.1
    )
    
    # Create dummy vocabulary
    vocab = Vocabulary(min_freq=1, max_vocab_size=vocab_size)
    vocab.add_sentences(["hello world", "test sentence"])
    vocab.build_vocab()
    
    # Create trainer
    trainer = create_trainer(
        model=model,
        vocabulary=vocab,
        learning_rate=1e-4,
        scheduler_type="cosine"
    )
    
    print(f"Trainer created successfully")
    print(f"Model parameters: {sum(p.numel() for p in trainer.model.parameters()):,}")
    print(f"Vocabulary size: {len(vocab)}")
    print(f"Device: {trainer.device}")
    
    print("\nTrainer test completed successfully!")
