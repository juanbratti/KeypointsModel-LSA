"""
Evaluation Metrics for Sign Language Translation

This module implements evaluation metrics including BLEU scores (1-4),
accuracy for sign language translation models.
"""

import torch
import numpy as np
from typing import List, Dict, Tuple, Optional, Any
from collections import Counter
import math
import re
from nltk.translate.bleu_score import sentence_bleu, corpus_bleu, SmoothingFunction

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


def tokenize_text(text: str) -> List[str]:
    """
    Simple tokenization for evaluation
    
    Args:
        text: Input text string
        
    Returns:
        List of tokens
    """
    # Convert to lowercase and split on whitespace and punctuation
    text = text.lower().strip()
    tokens = re.findall(r'\b\w+\b', text)
    return tokens


def compute_bleu_score(reference_tokens, candidate_tokens, 
                       weights=[0.25,0.25,0.25,0.25], smoothing=True) -> float:
    """
    Compute BLEU score for a single reference-candidate pair using NLTK
    """
    if not candidate_tokens:
        return 0.0
    smoothing_fn = SmoothingFunction().method1 if smoothing else None
    return sentence_bleu([reference_tokens], candidate_tokens,
                         weights=weights,
                         smoothing_function=smoothing_fn)

def compute_bleu_corpus(references, candidates, 
                        weights=[0.25,0.25,0.25,0.25], smoothing=True) -> float:
    """
    Compute corpus-level BLEU score using NLTK
    """
    assert len(references) == len(candidates), "Number of references and candidates must match"
    smoothing_fn = SmoothingFunction().method1 if smoothing else None
    # NLTK espera [[refs], [refs], ...], donde cada refs puede ser lista de múltiples refs
    refs_formatted = [[r] for r in references]
    return corpus_bleu(refs_formatted, candidates,
                       weights=weights,
                       smoothing_function=smoothing_fn)

def compute_individual_bleu_scores(references, candidates, smoothing=True):
    """
    Compute BLEU-1, BLEU-2, BLEU-3, and BLEU-4 using NLTK
    """
    ref_tokens = [tokenize_text(r) for r in references]
    cand_tokens = [tokenize_text(c) for c in candidates]
    bleu_scores = {}
    bleu_scores['BLEU-1'] = compute_bleu_corpus(ref_tokens, cand_tokens, weights=[1.0], smoothing=smoothing)
    bleu_scores['BLEU-2'] = compute_bleu_corpus(ref_tokens, cand_tokens, weights=[0.5,0.5], smoothing=smoothing)
    bleu_scores['BLEU-3'] = compute_bleu_corpus(ref_tokens, cand_tokens, weights=[1/3,1/3,1/3], smoothing=smoothing)
    bleu_scores['BLEU-4'] = compute_bleu_corpus(ref_tokens, cand_tokens, weights=[0.25,0.25,0.25,0.25], smoothing=smoothing)
    return bleu_scores


def compute_accuracy(references: List[str], 
                    candidates: List[str],
                    level: str = "sentence") -> float:
    """
    Compute accuracy at sentence or word level
    
    Args:
        references: List of reference texts
        candidates: List of candidate texts
        level: "sentence" or "word" level accuracy
        
    Returns:
        Accuracy score (0-1)
    """
    assert len(references) == len(candidates), "Number of references and candidates must match"
    
    if level == "sentence":
        # Sentence-level exact match accuracy
        correct = 0
        for ref, cand in zip(references, candidates):
            ref_clean = ref.strip().lower()
            cand_clean = cand.strip().lower()
            if ref_clean == cand_clean:
                correct += 1
        
        return correct / len(references)
    
    elif level == "word":
        # Word-level accuracy
        total_words = 0
        correct_words = 0
        
        for ref, cand in zip(references, candidates):
            ref_tokens = tokenize_text(ref)
            cand_tokens = tokenize_text(cand)
            
            # Align tokens and count matches
            max_len = max(len(ref_tokens), len(cand_tokens))
            
            for i in range(max_len):
                total_words += 1
                ref_token = ref_tokens[i] if i < len(ref_tokens) else ""
                cand_token = cand_tokens[i] if i < len(cand_tokens) else ""
                
                if ref_token == cand_token:
                    correct_words += 1
        
        return correct_words / total_words if total_words > 0 else 0.0
    
    else:
        raise ValueError(f"Unknown accuracy level: {level}")


class SignLanguageEvaluator:
    """
    Comprehensive evaluator for sign language translation
    """
    
    def __init__(self, vocabulary=None):
        """
        Initialize evaluator
        
        Args:
            vocabulary: Vocabulary for token conversion (optional)
        """
        self.vocabulary = vocabulary
    
    def evaluate_model(self, 
                      model,
                      data_loader,
                      device: torch.device,
                      max_samples: Optional[int] = None,
                      config: Optional[dict] = None) -> Dict[str, float]:
        """
        Evaluate model on dataset
        
        Args:
            model: Trained model
            data_loader: Data loader for evaluation
            device: Device for computation
            max_samples: Maximum number of samples to evaluate (None for all)
            
        Returns:
            Dictionary of evaluation metrics
        """
        model.eval()
        
        all_references = []
        all_candidates = []
        total_loss = 0.0
        total_samples = 0
        
        print("Iniciando evaluación del modelo...")
        print(f"Total de batches a procesar: {len(data_loader)}")
        if max_samples:
            print(f"Máximo de muestras: {max_samples}")
        
        with torch.no_grad():
            for batch_idx, batch in enumerate(data_loader):
                if max_samples and total_samples >= max_samples:
                    break
                
                # Progreso cada 10 batches
                if batch_idx % 10 == 0:
                    print(f"  Procesando batch {batch_idx + 1}/{len(data_loader)} (muestras: {total_samples})")
                
                # Extract data from the batch dictionary
                keypoints = batch['keypoints'].to(device)
                attention_masks = batch['attention_masks'].to(device)
                target_texts = batch['transcriptions']
                batch_size = keypoints.size(0)
                
                # VERIFICACIÓN OBLIGATORIA: dimensiones de entrada
                assert keypoints.size(-1) == 1086, f"OBLIGATORIO: dimensiones de entrada deben ser 1086, encontrado: {keypoints.size(-1)}"
                
                # Generate predictions using temperature sampling
                from inference.generator import SignLanguageGenerator
                generator = SignLanguageGenerator(model, self.vocabulary, device)
                
                # Generate predictions using beam search
                predicted_texts = []
                if batch_idx % 10 == 0:
                    print(f"    Generando predicciones para batch {batch_idx + 1} ({batch_size} muestras)...")
                
                for i in range(batch_size):
                    # Extract single sample
                    single_keypoints = keypoints[i:i+1]  # Keep batch dimension
                    
                    # Usar parámetros de config para generación
                    generation_config = config.get('generation', {}) if config else {}
                    max_length = generation_config.get('max_length', 60)
                    beam_size = generation_config.get('beam_size', 32)
                    method = generation_config.get('method', 'beam_search')  # Por defecto beam_search
                    
                    # Generar predicciones según el método configurado
                    if method == 'greedy':
                        results = generator.generate_greedy(
                            single_keypoints,
                            max_length=max_length
                        )
                    else:  # beam_search
                        results = generator.generate_beam_search(
                            single_keypoints, 
                            beam_size=beam_size,
                            max_length=max_length
                        )
                    predicted_texts.extend(results['texts'])
                
                if batch_idx % 10 == 0:
                    print(f"    Predicciones generadas para batch {batch_idx + 1}")
                
                # Add to evaluation lists
                if isinstance(target_texts[0], str):
                    all_references.extend(target_texts)
                else:
                    # Convert token indices to text
                    all_references.extend(
                        self.vocabulary.decode_batch(target_texts, remove_special_tokens=True)
                    )
                
                all_candidates.extend(predicted_texts)
                total_samples += batch_size
                
                # Mostrar ejemplo cada 50 batches
                if batch_idx % 50 == 0 and batch_idx > 0:
                    print(f"    Ejemplo (batch {batch_idx}):")
                    print(f"      Ref:  '{target_texts[0][:80]}...'")
                    print(f"      Pred: '{predicted_texts[0][:80]}...'")
                    print(f"    Progreso: {total_samples} muestras procesadas")
                
                # Compute loss if possible
                if self.vocabulary is not None:
                    target_tokens = self.vocabulary.encode_batch(
                        target_texts if isinstance(target_texts[0], str) else 
                        self.vocabulary.decode_batch(target_texts),
                        add_special_tokens=True,
                        padding=True
                    ).to(device)
                    
                    # Teacher forcing forward pass for loss computation
                    tgt_input = target_tokens[:, :-1]
                    tgt_output = target_tokens[:, 1:]
                    
                    # Convert attention_mask to src_key_padding_mask format
                    src_key_padding_mask = (attention_masks == 0) if attention_masks is not None else None
                    tgt_key_padding_mask = (tgt_input == self.vocabulary.PAD_IDX)
                    
                    vocab_probs = model(keypoints, tgt_input, 
                                      src_key_padding_mask=src_key_padding_mask,
                                      tgt_key_padding_mask=tgt_key_padding_mask,
                                      memory_key_padding_mask=src_key_padding_mask)
                    
                    # Compute loss
                    criterion = torch.nn.CrossEntropyLoss(ignore_index=self.vocabulary.PAD_IDX)
                    loss = criterion(
                        vocab_probs.reshape(-1, vocab_probs.size(-1)),
                        tgt_output.reshape(-1)
                    )
                    total_loss += loss.item() * batch_size
        
        print(f"--------- Evaluación completada")
        print(f"--------- Total de muestras procesadas: {total_samples}")
        print(f"--------- Calculando métricas...")
        
        # Compute metrics
        metrics = {}
        
        # BLEU scores
        print("--------- Calculando BLEU scores...")
        bleu_scores = compute_individual_bleu_scores(all_references, all_candidates)
        metrics.update(bleu_scores)
        
        # Accuracy
        print("---------Calculando precisión...")
        metrics['Sentence_Accuracy'] = compute_accuracy(all_references, all_candidates, "sentence")
        metrics['Word_Accuracy'] = compute_accuracy(all_references, all_candidates, "word")
        
        
        # Additional statistics
        metrics['Total_Samples'] = total_samples
        
        print("--------- Métricas calculadas exitosamente!")
        return metrics
    
    def print_evaluation_results(self, metrics: Dict[str, float]):
        """
        Print evaluation results in a formatted way
        
        Args:
            metrics: Dictionary of evaluation metrics
        """
        print("\n" + "="*60)
        print("RESULTADOS DE EVALUACIÓN FINAL")
        print("="*60)
        
        # BLEU scores
        print("BLEU Scores:")
        for bleu_type in ['BLEU-1', 'BLEU-2', 'BLEU-3', 'BLEU-4']:
            if bleu_type in metrics:
                print(f"  {bleu_type}: {metrics[bleu_type]:.4f}")
        
        # Accuracy
        print("\nAccuracy:")
        if 'Sentence_Accuracy' in metrics:
            print(f"  Sentence: {metrics['Sentence_Accuracy']:.4f}")
        if 'Word_Accuracy' in metrics:
            print(f"  Word: {metrics['Word_Accuracy']:.4f}")
        
        
        # Sample count
        if 'Total_Samples' in metrics:
            print(f"\nTotal Samples: {metrics['Total_Samples']}")
        
        print("=" * 50)

