"""
Metrics computation for VQA evaluation.
Contains all metric calculation logic (BLEU, METEOR, ROUGE, CIDEr, BERTScore, Accuracy).
Uses underthesea for Vietnamese text preprocessing (normalization + word segmentation).
"""

import re
import unicodedata
from typing import Dict, List, Tuple
import torch
from pycocoevalcap.bleu.bleu import Bleu
from pycocoevalcap.meteor.meteor import Meteor
from pycocoevalcap.rouge.rouge import Rouge
from pycocoevalcap.cider.cider import Cider
from torchmetrics.text import BERTScore
from underthesea import text_normalize, word_tokenize
import os

# ============================================================================
# SHARED RESOURCES
# ============================================================================

class SharedBERTScoreModel:
    """Singleton for shared BERTScore model to avoid repeated initialization."""
    
    _instance = None
    _device = None
    _model_path = None
    
    HF_MODEL_NAME = "vinai/phobert-base"
    LOCAL_CACHE_DIR = "/home/research/workspace/model/phobert-base"
    
    @classmethod
    def get_model_path(cls) -> str:
        """Get model path: use local cache if exists, otherwise use HuggingFace."""
        if os.path.exists(cls.LOCAL_CACHE_DIR):
            config_file = os.path.join(cls.LOCAL_CACHE_DIR, "config.json")
            has_model = (
                os.path.exists(os.path.join(cls.LOCAL_CACHE_DIR, "pytorch_model.bin")) or 
                os.path.exists(os.path.join(cls.LOCAL_CACHE_DIR, "model.safetensors"))
            )
            
            if os.path.exists(config_file) and has_model:
                return cls.LOCAL_CACHE_DIR
        
        os.makedirs(cls.LOCAL_CACHE_DIR, exist_ok=True)
        return cls.HF_MODEL_NAME
    
    @classmethod
    def get_instance(cls, device: str = "cuda") -> BERTScore:
        """Get or initialize shared BERTScore model."""
        if cls._instance is None or cls._device != device:
            cls._device = device
            cls._model_path = cls.get_model_path()
            
            cls._instance = BERTScore(
                model_name_or_path=cls._model_path,
                num_layers=12,
                rescale_with_baseline=False,
                device=device,
                truncation=True,
                max_length=256,
                dist_sync_on_step=False,
                sync_on_compute=False
            )
        
        return cls._instance


# ============================================================================
# TEXT PREPROCESSING
# ============================================================================

def clean_text(text: str) -> str:
    """
    Remove line breaks, control characters, and normalize whitespace.
    
    Args:
        text: Raw input text
        
    Returns:
        Cleaned text with normalized whitespace
    """
    if not text:
        return ""
    
    # Replace various line breaks and separators
    text = text.replace("|||", " ").replace("\r\n", " ").replace("\n", " ").replace("\r", " ")
    
    # Remove control characters
    text = "".join(ch for ch in text if unicodedata.category(ch) != "Cc")
    
    # Normalize whitespace
    return re.sub(r"\s+", " ", text).strip()


def normalize_answer(text: str) -> str:
    """
    Normalize answer for exact matching.
    Applies: lowercase, punctuation removal, word sorting.
    
    Args:
        text: Raw answer text
        
    Returns:
        Normalized answer string
    """
    if not text:
        return ""
    
    text = clean_text(text).lower().strip().rstrip(".").replace('"', "").strip()
    
    # Yes/No normalization for Vietnamese and English
    if text in ["có", "đúng", "vâng", "yes", "true", "correct"]:
        return "có"
    if text in ["không", "sai", "no", "false", "incorrect"]:
        return "không"
    
    # Remove special characters and normalize whitespace
    text = re.sub(r'[^\w\s]', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    
    # Sort words for order-independent matching
    return " ".join(sorted(text.split()))


def normalize_explanation(text: str) -> str:
    """
    Normalize explanation text.
    Removes 'because'/'vì' prefix and trailing punctuation.
    
    Args:
        text: Raw explanation text
        
    Returns:
        Normalized explanation
    """
    text = clean_text(text).strip().rstrip(".").strip()
    
    # Remove 'because'/'vì' prefix
    text_lower = text.lower()
    if text_lower.startswith("because "):
        text = text[8:].strip()
    elif text_lower.startswith("vì "):
        text = text[3:].strip()
    
    return text


def truncate_text(text: str, max_words: int) -> str:
    """
    Truncate text to maximum number of words.
    
    Args:
        text: Input text
        max_words: Maximum word count
        
    Returns:
        Truncated text
    """
    words = text.split()
    return " ".join(words[:max_words]) if len(words) > max_words else text


def preprocess_vietnamese_text(text: str) -> str:
    """
    Complete Vietnamese text preprocessing pipeline using underthesea.
    
    Pipeline:
    1. Text normalization (fix encoding, typos)
    2. Word segmentation (tokenization)
    
    Args:
        text: Raw Vietnamese text
        
    Returns:
        Preprocessed text ready for PhoBERT
        
    Example:
        Input:  "Ðảm baỏ chất lựơng phòng thí nghịêm hoá học"
        After normalize: "Đảm bảo chất lượng phòng thí nghiệm hóa học"
        After tokenize: "Đảm_bảo chất_lượng phòng thí_nghiệm hóa_học"
    """
    if not text or not text.strip():
        return ""
    
    try:
        # Step 1: Normalize text (fix Vietnamese encoding issues and typos)
        normalized_text = text_normalize(text)
        
        # Step 2: Word tokenization with underscore format for PhoBERT
        # format="text" returns "word1_word2 word3" format
        tokenized_text = word_tokenize(normalized_text, format="text")
        
        return tokenized_text
    except Exception as e:
        print(f"   ⚠️ Preprocessing error: {e}")
        return text


# ============================================================================
# ACCURACY COMPUTATION
# ============================================================================

def compute_accuracy(predictions: List[str], ground_truths: List[str]) -> Tuple[float, int, int]:
    """
    Compute exact match accuracy between predictions and ground truths.
    
    Args:
        predictions: List of predicted answers
        ground_truths: List of ground truth answers
        
    Returns:
        Tuple of (accuracy_percentage, correct_count, total_count)
    """
    if not predictions or len(predictions) != len(ground_truths):
        return 0.0, 0, 0
    
    correct = sum(1 for pred, gt in zip(predictions, ground_truths) 
                  if normalize_answer(pred) == normalize_answer(gt))
    total = len(predictions)
    accuracy = (correct / total * 100) if total > 0 else 0.0
    
    return accuracy, correct, total


# ============================================================================
# NLG METRICS (BLEU, METEOR, ROUGE, CIDEr)
# ============================================================================

def compute_traditional_metrics(gts: Dict[int, List[str]], res: Dict[int, List[str]]) -> Dict[str, float]:
    """
    Compute BLEU, METEOR, ROUGE, CIDEr scores.
    
    Args:
        gts: Ground truths - {img_id: [reference_captions]}
        res: Predictions - {img_id: [predicted_caption]}
        
    Returns:
        Dictionary of metric scores (scaled to 0-100)
    """
    scorers = [
        (Bleu(4), ["BLEU-1", "BLEU-2", "BLEU-3", "BLEU-4"]),
        (Meteor(), "METEOR"),
        (Rouge(), "ROUGE_L"),
        (Cider(), "CIDEr"),
    ]
    
    scores = {}
    for scorer, method in scorers:
        try:
            score, _ = scorer.compute_score(gts, res)
            if isinstance(method, list):
                for m, s in zip(method, score):
                    scores[m] = float(s) * 100
            else:
                scores[method] = float(score) * 100
        except Exception as e:
            print(f"   ⚠️ Error computing {method}: {e}")
            if isinstance(method, list):
                scores.update({m: 0.0 for m in method})
            else:
                scores[method] = 0.0
    
    return scores


# ============================================================================
# BERTSCORE COMPUTATION
# ============================================================================

def compute_bertscore_single(candidate: str, reference: str, device: str = "cuda") -> float:
    """
    Compute BERTScore F1 for a single candidate-reference pair.
    
    Args:
        candidate: Predicted text
        reference: Ground truth text
        device: cuda or cpu
        
    Returns:
        BERTScore F1 score (0-100)
    """
    if not candidate.strip() or not reference.strip():
        return 0.0
    
    try:
        bertscore = SharedBERTScoreModel.get_instance(device=device)
        bertscore.reset()
        bertscore.update([candidate], [reference])
        f1_score = bertscore.compute()['f1']
        
        # Handle both 0-dim and 1-dim tensors
        return f1_score.item() * 100 if f1_score.dim() == 0 else f1_score[0].item() * 100
    except Exception as e:
        print(f"   ⚠️ BERTScore error: {e}")
        return 0.0


def compute_bertscore_max_ref(hypotheses: List[str], references: List[List[str]], 
                              device: str = "cuda", batch_size: int = 8) -> List[float]:
    """
    Compute BERTScore F1 with max over multiple references for each hypothesis.
    Uses batch processing for efficiency.
    
    Args:
        hypotheses: List of predicted texts
        references: List of reference lists (multiple refs per hypothesis)
        device: cuda or cpu
        batch_size: Batch size for BERTScore computation
        
    Returns:
        List of max BERTScore F1 scores (0-100)
    """
    if not hypotheses:
        return []
    
    # Prepare all candidate-reference pairs
    all_candidates = []
    all_references = []
    pair_indices = []  # Track which hypothesis each pair belongs to
    
    for idx, (hyp, refs) in enumerate(zip(hypotheses, references)):
        valid_refs = [r for r in refs if r.strip()]
        
        if not hyp.strip() or not valid_refs:
            continue
        
        # Create pairs: repeat hypothesis for each reference
        for ref in valid_refs:
            all_candidates.append(hyp)
            all_references.append(ref)
            pair_indices.append(idx)
    
    if not all_candidates:
        return [0.0] * len(hypotheses)
    
    # Compute BERTScore in batches
    bertscore = SharedBERTScoreModel.get_instance(device=device)
    bertscore.reset()
    
    # Process in batches
    all_scores = []
    for i in range(0, len(all_candidates), batch_size):
        batch_cands = all_candidates[i:i + batch_size]
        batch_refs = all_references[i:i + batch_size]
        
        bertscore.update(batch_cands, batch_refs)
        batch_result = bertscore.compute()
        
        # Extract F1 scores and convert to list
        f1_scores = batch_result['f1'].cpu().tolist()
        if isinstance(f1_scores, float):
            f1_scores = [f1_scores]
        
        all_scores.extend(f1_scores)
        bertscore.reset()
    
    # Group scores by hypothesis and take max
    max_scores = [0.0] * len(hypotheses)
    for pair_idx, score in zip(pair_indices, all_scores):
        max_scores[pair_idx] = max(max_scores[pair_idx], score * 100)
    
    return max_scores


# ============================================================================
# COMBINED NLG METRICS
# ============================================================================

def compute_all_nlg_metrics(references: List[List[str]], hypotheses: List[str], 
                            device: str = "cuda", max_len: int = 150) -> Dict[str, float]:
    """
    Compute all NLG metrics (BLEU, METEOR, ROUGE, CIDEr, BERTScore).
    
    Pipeline:
    1. Truncate texts to max_len words
    2. Preprocess with underthesea (normalize + tokenize)
    3. Compute traditional metrics (BLEU, METEOR, ROUGE, CIDEr)
    4. Compute BERTScore with max over references
    
    Args:
        references: List of reference lists (multiple refs per sample)
        hypotheses: List of predictions
        device: cuda or cpu
        max_len: Maximum words per text
        
    Returns:
        Dictionary of all metric scores (0-100 scale)
    """
    # Step 1: Truncate texts
    hypotheses = [truncate_text(h, max_len) for h in hypotheses]
    references = [[truncate_text(r, max_len) for r in refs] for refs in references]
    
    # Step 2: Preprocess with underthesea (normalize + tokenize)
    hypotheses = [preprocess_vietnamese_text(h) for h in hypotheses]
    references = [[preprocess_vietnamese_text(r) for r in refs] for refs in references]
    
    # Step 3: Prepare dictionaries for traditional metrics
    gts = {i: [clean_text(r) for r in refs] for i, refs in enumerate(references)}
    res = {i: [clean_text(hyp)] for i, hyp in enumerate(hypotheses)}
    
    # Step 4: Compute traditional metrics
    scores = compute_traditional_metrics(gts, res)
    
    # Step 5: Compute BERTScore
    print("   ⏳ Computing BERTScore...")
    max_f1_scores = compute_bertscore_max_ref(hypotheses, references, device)
    scores["BERTScore_F1"] = sum(max_f1_scores) / len(max_f1_scores) if max_f1_scores else 0.0
    print("   ✅ BERTScore complete")
    
    return scores
