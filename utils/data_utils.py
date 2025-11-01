"""
ARCHIVO NO UTILIZADO - Solo para referencia
    El proyecto principal usa lsa_dataset.py con secuencias completas.

Data utilities for sign language keypoint processing

This module provides helper functions for loading, preprocessing,
and visualizing keypoint data for sign language translation.

Learning Focus:
- Understanding data preprocessing pipelines
- Working with HDF5 files
- Keypoint normalization techniques
"""

import h5py
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from typing import Tuple, Optional, List
import matplotlib.pyplot as plt


class KeypointDataset(Dataset):
    """
    PyTorch Dataset for keypoint-based sign language data
    
    This is a learning-oriented implementation with extensive comments
    to help understand how PyTorch datasets work.
    """
    
    def __init__(self, hdf5_path: str, split: str = 'train', 
                 max_sequence_length: int = 100, normalize: bool = True):
        """
        Initialize the dataset
        
        Args:
            hdf5_path: Path to the HDF5 file containing keypoint data
            split: Data split ('train', 'val', 'test')
            max_sequence_length: Maximum length of keypoint sequences
            normalize: Whether to normalize keypoint coordinates
        """
        self.hdf5_path = hdf5_path
        self.split = split
        self.max_sequence_length = max_sequence_length
        self.normalize = normalize
        
        # Load data indices and metadata
        self._load_metadata()
        
    def _load_metadata(self):
        """Load dataset metadata without loading all data into memory"""
        with h5py.File(self.hdf5_path, 'r') as f:
            # TODO: Update these keys based on your actual HDF5 structure
            # Common structures might be:
            # - f[f'{self.split}/keypoints'] for keypoint data
            # - f[f'{self.split}/labels'] for text labels
            # - f[f'{self.split}/metadata'] for additional info
            
            # Example structure - update based on your data:
            if f'{self.split}/keypoints' in f:
                self.keypoints_key = f'{self.split}/keypoints'
                self.labels_key = f'{self.split}/labels'
                self.num_samples = f[self.keypoints_key].shape[0]
            else:
                # Fallback for different structure
                self.keypoints_key = 'keypoints'
                self.labels_key = 'labels'
                self.num_samples = f[self.keypoints_key].shape[0]
                
        print(f"Loaded {self.split} split with {self.num_samples} samples")
    
    def __len__(self) -> int:
        """Return the number of samples in the dataset"""
        return self.num_samples
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, str]:
        """
        Get a single sample from the dataset
        
        Args:
            idx: Sample index
            
        Returns:
            Tuple of (keypoints_tensor, label_text)
        """
        with h5py.File(self.hdf5_path, 'r') as f:
            # Load keypoints for this sample
            keypoints = f[self.keypoints_key][idx]  # Shape: (seq_len, num_keypoints, 2)
            
            # Load corresponding label/text
            if self.labels_key in f:
                label = f[self.labels_key][idx]
                if isinstance(label, bytes):
                    label = label.decode('utf-8')
            else:
                label = f"sample_{idx}"  # Fallback label
        
        # Preprocess keypoints
        keypoints = self._preprocess_keypoints(keypoints)
        
        # Convert to PyTorch tensors
        keypoints_tensor = torch.FloatTensor(keypoints)
        
        return keypoints_tensor, label
    
    def _preprocess_keypoints(self, keypoints: np.ndarray) -> np.ndarray:
        """
        Preprocess keypoint data
        
        Args:
            keypoints: Raw keypoint data, shape (seq_len, num_keypoints, 2)
            
        Returns:
            Preprocessed keypoints
        """
        # Handle sequence length
        seq_len = keypoints.shape[0]
        if seq_len > self.max_sequence_length:
            # Truncate long sequences
            keypoints = keypoints[:self.max_sequence_length]
        elif seq_len < self.max_sequence_length:
            # Pad short sequences with zeros
            padding = np.zeros((self.max_sequence_length - seq_len, 
                              keypoints.shape[1], keypoints.shape[2]))
            keypoints = np.concatenate([keypoints, padding], axis=0)
        
        # Normalize keypoints if requested
        if self.normalize:
            keypoints = self._normalize_keypoints(keypoints)
            
        return keypoints
    
    def _normalize_keypoints(self, keypoints: np.ndarray) -> np.ndarray:
        """
        Normalize keypoint coordinates
        
        Common normalization strategies:
        1. Center around a reference point (e.g., neck/torso)
        2. Scale by body size or frame dimensions
        3. Z-score normalization
        
        Args:
            keypoints: Keypoint data, shape (seq_len, num_keypoints, 2)
            
        Returns:
            Normalized keypoints
        """
        # Simple normalization: center around mean and scale by std
        # This is a basic approach - you might want more sophisticated methods
        
        # Remove frames where all keypoints are zero (padding)
        non_zero_mask = np.any(keypoints.reshape(keypoints.shape[0], -1) != 0, axis=1)
        
        if np.any(non_zero_mask):
            valid_keypoints = keypoints[non_zero_mask]
            
            # Center around mean position
            mean_pos = np.mean(valid_keypoints, axis=(0, 1), keepdims=True)
            keypoints[non_zero_mask] -= mean_pos
            
            # Scale by standard deviation
            std_pos = np.std(valid_keypoints) + 1e-8  # Add small epsilon to avoid division by zero
            keypoints[non_zero_mask] /= std_pos
        
        return keypoints


def create_data_loaders(hdf5_path: str, batch_size: int = 32, 
                       num_workers: int = 4) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """
    Create PyTorch DataLoaders for train, validation, and test sets
    
    Args:
        hdf5_path: Path to HDF5 dataset file
        batch_size: Batch size for training
        num_workers: Number of workers for data loading
        
    Returns:
        Tuple of (train_loader, val_loader, test_loader)
    """
    # Create datasets
    train_dataset = KeypointDataset(hdf5_path, split='train')
    val_dataset = KeypointDataset(hdf5_path, split='val')
    test_dataset = KeypointDataset(hdf5_path, split='test')
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset, 
        batch_size=batch_size, 
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True  # Faster GPU transfer
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )
    
    return train_loader, val_loader, test_loader


def visualize_keypoint_sequence(keypoints: np.ndarray, title: str = "Keypoint Sequence"):
    """
    Visualize a sequence of keypoints over time
    
    Args:
        keypoints: Keypoint sequence, shape (seq_len, num_keypoints, 2)
        title: Plot title
    """
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    axes = axes.flatten()
    
    # Show keypoints at different time steps
    time_steps = np.linspace(0, len(keypoints)-1, 6, dtype=int)
    
    for i, t in enumerate(time_steps):
        ax = axes[i]
        frame_keypoints = keypoints[t]
        
        # Plot keypoints
        x_coords = frame_keypoints[:, 0]
        y_coords = frame_keypoints[:, 1]
        
        ax.scatter(x_coords, y_coords, c='red', s=20, alpha=0.7)
        ax.set_title(f"Frame {t}")
        ax.set_xlabel("X coordinate")
        ax.set_ylabel("Y coordinate")
        ax.grid(True, alpha=0.3)
        ax.axis('equal')
    
    plt.suptitle(title)
    plt.tight_layout()
    plt.show()


def print_dataset_statistics(hdf5_path: str):
    """
    Print useful statistics about the dataset
    
    Args:
        hdf5_path: Path to HDF5 dataset file
    """
    print("Dataset Statistics")
    print("=" * 50)
    
    try:
        with h5py.File(hdf5_path, 'r') as f:
            print("Dataset structure:")
            
            def print_info(name, obj):
                indent = "  " * (name.count('/'))
                if hasattr(obj, 'shape'):
                    print(f"{indent}{name}: {obj.shape} ({obj.dtype})")
                else:
                    print(f"{indent}{name}: {type(obj).__name__}")
            
            f.visititems(print_info)
            
    except FileNotFoundError:
        print(f"Dataset file not found: {hdf5_path}")
        print("Please check the file path and try again.")
    except Exception as e:
        print(f"Error reading dataset: {e}")


if __name__ == "__main__":
    # Example usage
    print("Data Utils Module")
    print("This module provides utilities for loading and preprocessing keypoint data.")
    print("\nTo use this module:")
    print("1. Update the KeypointDataset class with your HDF5 structure")
    print("2. Use create_data_loaders() to get PyTorch DataLoaders")
    print("3. Use visualization functions to explore your data")
