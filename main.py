"""
Main training script for Sign Language Translation

This script provides a complete training pipeline for the sign language
translation model, including data loading, model creation, training, and evaluation.

Usage:
    python main.py --config configs/lsa_t_config.yaml --mode train
    python main.py --config configs/lsa_t_config.yaml --mode evaluate
    python main.py --config configs/lsa_t_config.yaml --mode inference --input keypoints.npy
"""

import argparse
import yaml
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import logging
import os
from pathlib import Path
import numpy as np

# Local imports
from models.sign_language_translator import SignLanguageTranslator, create_model_from_config
from utils.vocabulary import Vocabulary, build_vocabulary_from_texts
from utils.data_utils import KeypointDataset, create_data_loaders
from training.trainer import SignLanguageTrainer, create_trainer
from evaluation.metrics import SignLanguageEvaluator
from inference.generator import SignLanguageGenerator


def setup_logging(config: dict):
    """Setup logging configuration"""
    log_level = getattr(logging, config['logging']['log_level'].upper())
    log_file = config['logging'].get('log_file', 'training.log')
    
    logging.basicConfig(
        level=log_level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )
    
    return logging.getLogger(__name__)


def load_config(config_path: str) -> dict:
    """Load configuration from YAML file"""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def create_directories(config: dict):
    """Create necessary directories"""
    for path_key, path_value in config['paths'].items():
        # Skip file paths, only create directories
        if path_key != 'vocab_path':  # vocab_path should be a file, not a directory
            Path(path_value).mkdir(parents=True, exist_ok=True)


def load_or_create_vocabulary(config: dict, train_texts: list = None) -> Vocabulary:
    """Load existing vocabulary or create new one"""
    vocab_path = config['paths']['vocab_path']
    
    if os.path.exists(vocab_path):
        print(f"Loading vocabulary from {vocab_path}")
        vocabulary = Vocabulary.load(vocab_path)
    else:
        if train_texts is None:
            raise ValueError("No training texts provided and no existing vocabulary found")
        
        print(f"Creating new vocabulary from {len(train_texts)} training texts...")
        vocabulary = build_vocabulary_from_texts(
            train_texts,
            min_freq=1,  # Include all words (freq >= 1)
            max_vocab_size=config['model']['vocab_size'],
            lowercase=True
        )
        
        # Save vocabulary (create parent directory if needed)
        vocab_dir = os.path.dirname(vocab_path)
        if vocab_dir:  # Only create directory if there is one
            os.makedirs(vocab_dir, exist_ok=True)
            print(f"Created vocabulary directory: {vocab_dir}")
        vocabulary.save(vocab_path)
        print(f"Vocabulary saved to {vocab_path}")
        print(f"Final vocabulary stats: {len(vocabulary)} words (min_freq=1, max_size={config['model']['vocab_size']})")
    
    return vocabulary


def create_model(config: dict, vocabulary: Vocabulary) -> SignLanguageTranslator:
    """Create and initialize model"""
    model_config = {
        'd_model': config['model']['d_model'],
        'encoder_layers': config['model']['encoder_layers'],
        'decoder_layers': config['model']['decoder_layers'],
        'num_heads': config['model']['num_heads'],
        'd_ff': config['model'].get('d_ff', config['model']['d_model'] * 4),
        'dropout': config['model']['dropout'],
        'max_seq_length': config['model']['max_seq_length'],
        'keypoint_dim': config['model']['keypoint_dim'],
        'pad_idx': vocabulary.PAD_IDX
    }
    
    model = create_model_from_config(model_config, len(vocabulary))
    
    # Print model info
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    print(f"Model Information:")
    print(f"  Total parameters: {total_params:,}")
    print(f"  Trainable parameters: {trainable_params:,}")
    print(f"  Model size: {total_params * 4 / 1024 / 1024:.2f} MB")
    
    return model


def train_model(config: dict, logger: logging.Logger):
    """Main training function"""
    logger.info("Starting training pipeline")
    
    # GPU check
    logger.info(f"Initial GPU check: CUDA available = {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        logger.info(f"GPU: {torch.cuda.get_device_name(0)}")
        logger.info(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
    
    # Create directories
    create_directories(config)
    
    # Load data
    logger.info("Loading training data...")
    
    # Import LSA dataset loader
    from utils.lsa_dataset import create_lsa_data_loaders, LSAKeypointDataset
    
    # Create data loaders for LSA dataset - UPDATED for variable length sequences
    train_loader, val_loader, test_loader = create_lsa_data_loaders(
        hdf5_path=config['data']['hdf5_path'],
        csv_path=config['data'].get('csv_path', 'data/meta.csv'),
        batch_size=config['data']['batch_size'],
        num_workers=config['data']['num_workers']
        # max_sequence_length removed - now uses variable length sequences
    )
    
    # Extract texts for vocabulary creation from ALL splits
    logger.info("Extracting texts for vocabulary creation from ALL dataset splits...")
    all_texts = []
    
    # Process all splits for complete vocabulary
    for split in ['train', 'val', 'test']:
        logger.info(f"Processing {split} split...")
        dataset = LSAKeypointDataset(
            config['data']['hdf5_path'],
            config['data'].get('csv_path', 'data/meta.csv'),
            split=split
        )
        
        logger.info(f"Processing ALL {len(dataset)} samples from {split} split...")
        for i in range(len(dataset)):
            if i % 1000 == 0:
                logger.info(f"  Processed {i}/{len(dataset)} samples from {split}...")
            try:
                _, transcription = dataset[i]
                all_texts.append(transcription)
            except Exception as e:
                logger.warning(f"  Skipped {split} sample {i} due to error: {e}")
                continue
    
    logger.info(f"Extracted {len(all_texts)} transcriptions from ALL splits for vocabulary")
    train_texts = all_texts
    
    # Load or create vocabulary
    logger.info("Creating vocabulary from training texts...")
    vocabulary = load_or_create_vocabulary(config, train_texts)
    logger.info(f"Vocabulary created successfully with {len(vocabulary)} words")
    logger.info(f"Special tokens: PAD={vocabulary.PAD_IDX}, UNK={vocabulary.UNK_IDX}, SOS={vocabulary.SOS_IDX}, EOS={vocabulary.EOS_IDX}")
    
    # Create model
    logger.info("Creating model architecture...")
    model = create_model(config, vocabulary)
    logger.info("Model created successfully")
    
    # Create trainer
    if config['hardware']['device'] == 'auto':
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else:
        device = torch.device(config['hardware']['device'])
    
    # Check GPU availability and memory
    if device.type == 'cuda':
        if torch.cuda.is_available():
            gpu_name = torch.cuda.get_device_name(0)
            gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1024**3
            logger.info(f"Using GPU: {gpu_name}")
            logger.info(f"GPU Memory: {gpu_memory:.1f} GB")
        else:
            logger.error("CUDA requested but not available! Falling back to CPU")
            device = torch.device('cpu')
    
    logger.info(f"Using device: {device}")
    logger.info("Creating trainer...")
    
    # Move model to device explicitly before creating trainer
    logger.info(f"Moving model to {device}...")
    model.to(device)
    logger.info(f"Model moved to {device}")
    
    # Check where model parameters are actually located
    first_param_device = next(model.parameters()).device
    logger.info(f"Model parameters are on: {first_param_device}")
    
    trainer = create_trainer(
        model=model,
        vocabulary=vocabulary,
        learning_rate=float(config['training']['learning_rate']),
        weight_decay=config['training']['weight_decay'],
        scheduler_type=config['scheduler']['type'],
        device=device,
        grad_clip=config['training']['grad_clip'],
        log_interval=config['training']['log_interval'],
        save_dir=config['paths']['checkpoint_dir']
    )
    logger.info("Trainer created successfully")
    
    # Double-check device after trainer creation
    trainer_param_device = next(trainer.model.parameters()).device
    logger.info(f"Trainer model parameters are on: {trainer_param_device}")
    
    # Train model
    logger.info("Starting training...")
    logger.info(f"Training configuration:")
    logger.info(f"  - Epochs: {config['training']['num_epochs']}")
    logger.info(f"  - Batch size: {config['data']['batch_size']}")
    logger.info(f"  - Learning rate: {config['training']['learning_rate']}")
    logger.info(f"  - Training samples: {len(train_loader.dataset)}")
    logger.info(f"  - Validation samples: {len(val_loader.dataset)}")
    
    history = trainer.train(
        train_loader=train_loader,
        val_loader=val_loader,
        num_epochs=config['training']['num_epochs'],
        save_every=config['training']['save_every'],
        early_stopping_patience=config['training']['patience']
    )
    
    logger.info("Training completed successfully!")
    
    # Save final model
    final_model_path = os.path.join(config['paths']['model_dir'], 'final_model.pt')
    torch.save({
        'model_state_dict': model.state_dict(),
        'config': config,
        'vocab_size': len(vocabulary)
    }, final_model_path)
    
    logger.info(f"Final model saved to {final_model_path}")


def evaluate_model(config: dict, logger: logging.Logger):
    """Evaluate trained model"""
    logger.info("Starting model evaluation")
    
    # Load vocabulary
    vocabulary = Vocabulary.load(config['paths']['vocab_path'])
    
    # Create model
    model = create_model(config, vocabulary)
    
    # Load trained weights
    checkpoint_path = os.path.join(config['paths']['checkpoint_dir'], 'best_model.pt')
    if os.path.exists(checkpoint_path):
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        model.load_state_dict(checkpoint['model_state_dict'])
        logger.info(f"Loaded model from {checkpoint_path}")
    else:
        logger.error(f"No trained model found at {checkpoint_path}")
        return
    
    # Setup device
    device = torch.device(config['hardware']['device'] if config['hardware']['device'] != 'auto' 
                         else 'cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    
    # VERIFICACIONES OBLIGATORIAS
    hdf5_path = config['data']['hdf5_path']
    assert hdf5_path == "keypoints_cleaned.h5", f"OBLIGATORIO: dataset debe ser keypoints_cleaned.h5, encontrado: {hdf5_path}"
    assert os.path.exists(hdf5_path), f"Dataset no encontrado: {hdf5_path}"
    
    # Load test data using LSA dataset loader
    try:
        from utils.lsa_dataset import create_lsa_data_loaders
        _, _, test_loader = create_lsa_data_loaders(
            hdf5_path=hdf5_path,
            csv_path=config['data'].get('csv_path', 'meta.csv'),
            batch_size=config['data']['batch_size'],
            num_workers=config['data']['num_workers']
        )
    except Exception as e:
        logger.error(f"Failed to load test data: {e}")
        return
    
    # Create evaluator
    evaluator = SignLanguageEvaluator(vocabulary)
    
    # Run evaluation with config
    metrics = evaluator.evaluate_model(model, test_loader, device, max_samples=1000, config=config)
    
    # Print results
    evaluator.print_evaluation_results(metrics)
    
    # Save results
    results_path = os.path.join(config['paths']['results_dir'], 'evaluation_results.yaml')
    with open(results_path, 'w') as f:
        yaml.dump(metrics, f)
    
    logger.info(f"Evaluation results saved to {results_path}")


def run_inference(config: dict, input_path: str, logger: logging.Logger):
    """Run inference on input keypoints"""
    logger.info(f"Running inference on {input_path}")
    
    # Load vocabulary
    vocabulary = Vocabulary.load(config['paths']['vocab_path'])
    
    # Create model
    model = create_model(config, vocabulary)
    
    # Load trained weights
    checkpoint_path = os.path.join(config['paths']['checkpoint_dir'], 'best_model.pt')
    if os.path.exists(checkpoint_path):
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        model.load_state_dict(checkpoint['model_state_dict'])
        logger.info(f"Loaded model from {checkpoint_path}")
    else:
        logger.error(f"No trained model found at {checkpoint_path}")
        return
    
    # Setup device
    device = torch.device(config['hardware']['device'] if config['hardware']['device'] != 'auto' 
                         else 'cuda' if torch.cuda.is_available() else 'cpu')
    
    # Load input keypoints
    try:
        if input_path.endswith('.npy'):
            keypoints = np.load(input_path)
        else:
            logger.error("Only .npy files are supported for keypoints")
            return
        
        # Convert to tensor
        keypoints_tensor = torch.FloatTensor(keypoints).unsqueeze(0)  # Add batch dimension
        
    except Exception as e:
        logger.error(f"Failed to load keypoints: {e}")
        return
    
    # Create generator
    generator = SignLanguageGenerator(model, vocabulary, device)
    
    # Generate translation
    print("Generating translation...")
    
    # Greedy decoding
    greedy_result = generator.generate_greedy(
        keypoints_tensor,
        max_length=config['generation']['max_length'],
    )
    
    print(f"Greedy translation: {greedy_result['texts'][0]}")
    
    # Beam search
    beam_result = generator.generate_beam_search(
        keypoints_tensor,
        beam_size=config['generation']['beam_size'],
        max_length=config['generation']['max_length'],
        length_penalty=config['generation']['length_penalty']
    )
    
    print(f"Beam search translation: {beam_result['texts'][0]}")
    
    logger.info("Inference completed successfully!")


def main():
    """Main function"""
    parser = argparse.ArgumentParser(description="Sign Language Translation")
    parser.add_argument('--config', type=str, required=True, 
                       help='Path to configuration file')
    parser.add_argument('--mode', type=str, choices=['train', 'evaluate', 'inference'],
                       default='train', help='Mode to run')
    parser.add_argument('--input', type=str, help='Input file for inference mode')
    
    args = parser.parse_args()
    
    # Load configuration
    config = load_config(args.config)
    
    # Setup logging
    logger = setup_logging(config)
    
    logger.info(f"Starting Sign Language Translation - Mode: {args.mode}")
    logger.info(f"Configuration: {args.config}")
    
    # Run appropriate mode
    if args.mode == 'train':
        train_model(config, logger)
    elif args.mode == 'evaluate':
        evaluate_model(config, logger)
    elif args.mode == 'inference':
        if not args.input:
            logger.error("Input file required for inference mode")
            return
        run_inference(config, args.input, logger)
    
    logger.info("Process completed!")


if __name__ == "__main__":
    main()
