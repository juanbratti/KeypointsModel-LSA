"""
LSA Dataset Loader for Sign Language Translation

This module provides a custom dataset loader for the LSA (Lengua de Señas Argentina)
dataset with HDF5 keypoints and CSV metadata structure.
"""

import h5py
import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from typing import Tuple, Optional, List, Dict, Any
import os
from pathlib import Path
import logging


class LSAKeypointDataset(Dataset):
    """
    Custom Dataset for LSA keypoint data with HDF5 structure:
    
    HDF5 Structure:
    ├── clip_1.mp4
    │   ├── signer_0
    │   │   ├── boxes: (frames, 4)
    │   │   └── keypoints: (frames, 2172)
    │   └── signer_1 (if exists)
    └── clip_N.mp4
    
    CSV Metadata:
    - id: Video clip identifier  
    - label: Text transcription
    - category: Thematic category
    - timing info: start, end, duration
    """
    
    def __init__(self,
                 hdf5_path: str,
                 csv_path: str,
                 split: str = 'train',
                 max_sequence_length: int = None,  # CAMBIADO: None = sin truncamiento
                 normalize: bool = True,
                 use_primary_signer: bool = True,
                 keypoint_subset: str = 'all',  # 'all', 'hands', 'face', 'body'
                 split_ratios: Tuple[float, float, float] = (0.8, 0.1, 0.1)):
        """
        Initialize LSA Dataset
        
        Args:
            hdf5_path: Path to HDF5 keypoints file
            csv_path: Path to CSV metadata file
            split: Dataset split ('train', 'val', 'test')
            max_sequence_length: Maximum sequence length
            normalize: Whether to normalize keypoints
            use_primary_signer: Whether to use only primary signer (signer_0)
            keypoint_subset: Which keypoints to use
            split_ratios: Train/val/test split ratios
        """
        self.hdf5_path = hdf5_path
        self.csv_path = csv_path
        self.split = split
        self.max_sequence_length = max_sequence_length
        self.normalize = normalize
        self.use_primary_signer = use_primary_signer
        self.keypoint_subset = keypoint_subset
        
        # Load metadata and create splits
        self.metadata_df = pd.read_csv(csv_path)
        
        # Use class-level cache for validation results
        if not hasattr(LSAKeypointDataset, '_validation_cache'):
            LSAKeypointDataset._validation_cache = {}
            
        self.clip_ids = self._create_splits(split_ratios)
        
        # Analyze keypoint structure
        self._analyze_keypoint_structure()
        
        print(f"Loaded {split} split with {len(self.clip_ids)} clips")
        print(f"Using keypoint subset: {keypoint_subset}")
        print(f"Primary signer only: {use_primary_signer}")
        if max_sequence_length is None:
            print(f"Sequence length: COMPLETAS (sin truncamiento)")
        else:
            print(f"Max sequence length: {max_sequence_length}")
        print(f"Normalization enabled: {normalize}")
    
    def _validate_clip(self, clip_id: str) -> bool:
        """Check if a clip has valid signer data (with caching)"""
        # Check cache first
        if clip_id in LSAKeypointDataset._validation_cache:
            return LSAKeypointDataset._validation_cache[clip_id]
            
        try:
            with h5py.File(self.hdf5_path, 'r') as f:
                # Try clip_id as is first, then with .mp4 extension
                if clip_id in f:
                    hdf5_key = clip_id
                elif f"{clip_id}.mp4" in f:
                    hdf5_key = f"{clip_id}.mp4"
                else:
                    LSAKeypointDataset._validation_cache[clip_id] = False
                    return False  # Clip not found in HDF5
                
                clip_group = f[hdf5_key]
                available_signers = list(clip_group.keys())
                
                # Check if has signers and at least one has keypoints
                if not available_signers:
                    LSAKeypointDataset._validation_cache[clip_id] = False
                    return False
                
                # Check if at least one signer has keypoints data
                for signer_key in available_signers:
                    if 'keypoints' in clip_group[signer_key]:
                        LSAKeypointDataset._validation_cache[clip_id] = True
                        return True
                
                LSAKeypointDataset._validation_cache[clip_id] = False
                return False
        except:
            LSAKeypointDataset._validation_cache[clip_id] = False
            return False
    
    def _create_splits(self, split_ratios: Tuple[float, float, float]) -> List[str]:
        """Create train/val/test splits from clip IDs"""
        # Get unique clip IDs
        all_clips = self.metadata_df['id'].unique().tolist()
        print(f"Validating {len(all_clips)} clips...")
        
        # Filter out clips without valid signer data
        valid_clips = []
        invalid_clips = []
        
        for i, clip_id in enumerate(all_clips):
            if i % 1000 == 0 and i > 0:
                print(f"  Validated {i}/{len(all_clips)} clips... ({len(valid_clips)} valid so far)")
            
            if self._validate_clip(clip_id):
                valid_clips.append(clip_id)
            else:
                invalid_clips.append(clip_id)
        
        print(f"Found {len(valid_clips)} valid clips")
        print(f"Skipped {len(invalid_clips)} clips without valid signers")
        
        # Always log skipped clips for reference
        if invalid_clips:
            skipped_log_path = f"skipped_clips_{self.split}.txt"
            with open(skipped_log_path, 'w') as f:
                f.write(f"Skipped clips in {self.split} split ({len(invalid_clips)} total):\n")
                f.write("=" * 50 + "\n")
                for clip in invalid_clips:
                    f.write(f"{clip}\n")
            print(f"Skipped clips logged to: {skipped_log_path}")
            
            # Show first few skipped clips
            print(f"First few skipped clips:")
            for clip in invalid_clips[:5]:  # Show first 5
                print(f"   • {clip}")
            if len(invalid_clips) > 5:
                print(f"   ... and {len(invalid_clips) - 5} more (see {skipped_log_path})")
        
        n_clips = len(valid_clips)
        
        # Calculate split sizes
        train_size = int(n_clips * split_ratios[0])
        val_size = int(n_clips * split_ratios[1])
        
        # Create splits (deterministic with seed)
        np.random.seed(42)  # For reproducible splits
        shuffled_clips = np.random.permutation(valid_clips)
        
        if self.split == 'train':
            return shuffled_clips[:train_size].tolist()
        elif self.split == 'val':
            return shuffled_clips[train_size:train_size + val_size].tolist()
        else:  # test
            return shuffled_clips[train_size + val_size:].tolist()
    
    def _analyze_keypoint_structure(self):
        """Analyze the keypoint structure to understand dimensions"""
        with h5py.File(self.hdf5_path, 'r') as f:
            # Get first available clip to analyze structure
            first_clip = list(f.keys())[0]
            first_signer = list(f[first_clip].keys())[0]
            
            # Load raw keypoints and apply x,y extraction
            raw_keypoints = f[f'{first_clip}/{first_signer}/keypoints'][:]
            keypoints = self._extract_xy_coordinates(raw_keypoints)
            
            print(f"Raw keypoints shape: {raw_keypoints.shape}")
            print(f"Processed keypoints shape: {keypoints.shape}")
            print(f"Total keypoint features (after x,y extraction): {keypoints.shape[1]}")
            print(f"Sample clip: {first_clip}")
            print(f"Sample signer: {first_signer}")
            
            self.total_features = keypoints.shape[1]  # Use processed dimensions
            
            # Define keypoint ranges based on MediaPipe structure (after x,y extraction)
            if self.total_features == 1086:
                print("Total features match expected 1086 (543 keypoints × 2 features)")
                # 543 keypoints × 2 features (x, y only)
                self.keypoint_ranges = {
                    'all': (0, 1086),
                    'pose': (0, 33 * 2),             # 0–65
                    'face': (33 * 2, 501 * 2),       # 66–1001
                    'left_hand': (501 * 2, 522 * 2), # 1002–1043
                    'right_hand': (522 * 2, 543 * 2),# 1044–1085
                    'hands': (501 * 2, 543 * 2)      # 1002–1085
                }
            else:
                print(f"Unexpected total features: {self.total_features}. Raising an error.")
                raise ValueError(f"Unexpected total features: {self.total_features}. Expected 1086.")
    
    def __len__(self) -> int:
        """Return number of clips in this split"""
        return len(self.clip_ids)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, str]:
        """
        Get a single sample
        
        Args:
            idx: Sample index
            
        Returns:
            Tuple of (keypoints_tensor, transcription_text)
        """
        clip_id = self.clip_ids[idx]
        
        # Load keypoints from HDF5
        keypoints = self._load_keypoints(clip_id)
        
        # Load transcription from CSV
        transcription = self._load_transcription(clip_id)
        
        # Preprocess keypoints
        keypoints = self._preprocess_keypoints(keypoints)
        
        # Convert to tensor
        keypoints_tensor = torch.FloatTensor(keypoints)
                
        return keypoints_tensor, transcription
    
    def _load_keypoints(self, clip_id: str) -> np.ndarray:
        """Load keypoints for a specific clip"""
        with h5py.File(self.hdf5_path, 'r') as f:
            # Try clip_id as is first, then with .mp4 extension
            if clip_id in f:
                hdf5_key = clip_id
            elif f"{clip_id}.mp4" in f:
                hdf5_key = f"{clip_id}.mp4"
            else:
                # Debug: show what keys are actually available
                available_keys = list(f.keys())[:10]  # Show first 10 keys
                raise ValueError(f"Clip {clip_id} (or {clip_id}.mp4) not found in HDF5 file. "
                               f"Available keys (first 10): {available_keys}")
            
            clip_group = f[hdf5_key]
            
            # Get available signers (should always exist since we pre-validated)
            available_signers = list(clip_group.keys())
            
            if self.use_primary_signer:
                # Use primary signer (signer_0)
                if 'signer_0' in clip_group:
                    signer_group = clip_group['signer_0']
                else:
                    # Fallback to first available signer
                    signer_key = available_signers[0]
                    signer_group = clip_group[signer_key]
                    print(f"Using fallback signer '{signer_key}' for clip {clip_id} (signer_0 not found)")
            else:
                if 'signer_0' in clip_group:
                    signer_group = clip_group['signer_0']
                else:
                    signer_key = available_signers[0]
                    signer_group = clip_group[signer_key]
                    print(f"Using fallback signer '{signer_key}' for clip {clip_id}")
            
            # Load keypoints
            keypoints = signer_group['keypoints'][:]  # Shape: (frames, 2172)
            
            # Extract only x,y coordinates (drop z and visibility)
            # 2172 features = 543 keypoints × 4 features (x, y, z, visibility)
            # We want 1086 features = 543 keypoints × 2 features (x, y)
            keypoints = self._extract_xy_coordinates(keypoints)
            
            # Extract subset of keypoints if specified
            if self.keypoint_subset in self.keypoint_ranges:
                # Use ranges directly (already adjusted for x,y coordinates in _analyze_keypoint_structure)
                start_idx, end_idx = self.keypoint_ranges[self.keypoint_subset]
                keypoints = keypoints[:, start_idx:end_idx]
            
            return keypoints
    
    def _extract_xy_coordinates(self, keypoints: np.ndarray) -> np.ndarray:
        """
        Extract only x,y coordinates from keypoints data.
        
        Input: (frames, 2172) where 2172 = 543 keypoints × 4 features (x, y, z, visibility)
        Output: (frames, 1086) where 1086 = 543 keypoints × 2 features (x, y)
        """
        frames, total_features = keypoints.shape
        
        if total_features != 2172:
            print(f"Expected 2172 features, got {total_features}. Using as-is.")
            return keypoints
        
        # Reshape to (frames, 543, 4) to separate keypoints and features
        keypoints_reshaped = keypoints.reshape(frames, 543, 4)
        
        # Extract only x,y coordinates (first 2 features)
        xy_keypoints = keypoints_reshaped[:, :, :2]  # (frames, 543, 2)
        
        # Reshape back to (frames, 1086)
        xy_keypoints_flat = xy_keypoints.reshape(frames, 543 * 2)
        
        return xy_keypoints_flat
    
    def _load_transcription(self, clip_id: str) -> str:
        """Load transcription for a specific clip"""
        # Find the row with matching clip_id
        clip_row = self.metadata_df[self.metadata_df['id'] == clip_id]
        
        if clip_row.empty:
            return f"No transcription for {clip_id}"
        
        # Get transcription text from 'label' column
        transcription = clip_row['label'].iloc[0]
        
        # Clean transcription if needed
        if isinstance(transcription, str):
            # Remove quotes and extra whitespace
            transcription = transcription.strip('"').strip("'").strip()
        else:
            transcription = str(transcription)
        
        return transcription
    
    def _preprocess_keypoints(self, keypoints: np.ndarray) -> np.ndarray:
        """
        Preprocess keypoints - MODIFIED for variable length sequences
        
        Args:
            keypoints: Raw keypoints (frames, features)
            
        Returns:
            Preprocessed keypoints with original length preserved
        """
        # Clean any NaN/Inf values in raw data
        if np.isnan(keypoints).any() or np.isinf(keypoints).any():
            keypoints = np.nan_to_num(keypoints, nan=0.0, posinf=1e3, neginf=-1e3)
        
        # Reshape keypoints based on your data structure
        frames, total_features = keypoints.shape
        
        if total_features % 4 == 0:
            # Assume 4 features per keypoint (x, y, z, visibility)
            num_keypoints = total_features // 4
            keypoints = keypoints.reshape(frames, num_keypoints, 4)
            
            # Drop Z and visibility, keep only X, Y
            keypoints = keypoints[:, :, :2]  # Shape: (frames, num_keypoints, 2)
            
            # Flatten to (frames, num_keypoints * 2)
            keypoints = keypoints.reshape(frames, -1)
        
        # Normalize if requested
        if self.normalize:
            keypoints = self._normalize_keypoints(keypoints)
        
        return keypoints
    
    def _normalize_keypoints(self, keypoints: np.ndarray) -> np.ndarray:
        """
        Per-frame, per-axis (object 2D) normalization.
        Leaves all-zero frames untouched.
        """
        T, F = keypoints.shape
        kp = keypoints.reshape(T, -1, 2).astype(np.float64)  # fuerza float64

        x, y = kp[..., 0], kp[..., 1]

        # detectar frames todos ceros
        zero_frames = (np.all(x == 0, axis=-1) & np.all(y == 0, axis=-1))

        norm_x, norm_y = np.zeros_like(x), np.zeros_like(y)

        for t in range(T):
            if zero_frames[t]:
                continue
            mean_x, std_x = x[t].mean(), x[t].std()
            mean_y, std_y = y[t].mean(), y[t].std()
            if std_x < 1e-8: std_x = 1.0
            if std_y < 1e-8: std_y = 1.0
            norm_x[t] = (x[t] - mean_x) / std_x
            norm_y[t] = (y[t] - mean_y) / std_y

        kp = np.stack([norm_x, norm_y], axis=-1)
        return kp.reshape(T, F).astype(np.float32)  # devolver en float32

def variable_length_collate_fn(batch):
    """
    Custom collate function for variable length sequences
    
    Args:
        batch: List of (keypoints_tensor, transcription) tuples
        
    Returns:
        Dictionary with padded keypoints, attention masks, and transcriptions
    """
    keypoints_list, transcriptions_list = zip(*batch)
    
    # Find the maximum sequence length in this batch
    max_length_in_batch = max(kp.shape[0] for kp in keypoints_list)
    batch_size = len(keypoints_list)
    keypoint_dim = keypoints_list[0].shape[1]  # Should be 1086
    
    # Initialize tensors
    padded_keypoints = torch.zeros(batch_size, max_length_in_batch, keypoint_dim)
    attention_masks = torch.zeros(batch_size, max_length_in_batch)
    sequence_lengths = []
    
    # Process each sequence in the batch
    for i, keypoints in enumerate(keypoints_list):
        seq_len = keypoints.shape[0]
        sequence_lengths.append(seq_len)
        
        # Copy the actual keypoints
        padded_keypoints[i, :seq_len, :] = keypoints
        
        # Create attention mask (1 for real frames, 0 for padding)
        attention_masks[i, :seq_len] = 1.0
    
    return {
        'keypoints': padded_keypoints,  # (batch_size, max_seq_len, 1086)
        'attention_masks': attention_masks,  # (batch_size, max_seq_len)
        'transcriptions': list(transcriptions_list),  # List of strings
        'sequence_lengths': sequence_lengths  # List of original lengths
    }


def create_lsa_data_loaders(hdf5_path: str,
                           csv_path: str,
                           batch_size: int = 32,
                           num_workers: int = 4,
                           max_sequence_length: int = None,  # Not used anymore
                           split_ratios: Tuple[float, float, float] = (0.8, 0.1, 0.1)) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """
    Create DataLoaders for LSA dataset
    
    Args:
        hdf5_path: Path to HDF5 keypoints file
        csv_path: Path to CSV metadata file
        batch_size: Batch size
        num_workers: Number of worker processes
        max_sequence_length: Maximum sequence length
        split_ratios: Train/val/test split ratios
        
    Returns:
        Tuple of (train_loader, val_loader, test_loader)
    """
    # Create datasets - no more max_sequence_length parameter
    train_dataset = LSAKeypointDataset(
        hdf5_path, csv_path, split='train',
        split_ratios=split_ratios
    )
    
    val_dataset = LSAKeypointDataset(
        hdf5_path, csv_path, split='val',
        split_ratios=split_ratios
    )
    
    test_dataset = LSAKeypointDataset(
        hdf5_path, csv_path, split='test',
        split_ratios=split_ratios
    )
    
    # Create data loaders with custom collate function
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        collate_fn=variable_length_collate_fn
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        collate_fn=variable_length_collate_fn
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        collate_fn=variable_length_collate_fn
    )
    
    return train_loader, val_loader, test_loader


if __name__ == "__main__":
    print("LSA Dataset module ready!")
