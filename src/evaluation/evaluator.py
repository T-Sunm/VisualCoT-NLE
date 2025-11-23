"""
VQA Evaluator class for loading predictions and computing evaluation metrics.
"""

import json
from typing import Dict, List, Any
from .metrics import ( 
    compute_accuracy,
    compute_all_nlg_metrics,
    normalize_explanation
)   


def ensure_list(value: Any) -> List[str]:
    """Convert value to list of strings."""
    if value is None:
        return []
    if isinstance(value, str):
        return [value]
    if isinstance(value, list):
        return [str(x) for x in value]
    return [str(value)]


class VQAEvaluator:
    """Main evaluator class for VQA predictions."""
    
    def __init__(self, device: str = "cuda"):
        """
        Initialize evaluator.
        
        Args:
            device: Device for BERTScore computation (cuda/cpu)
        """
        self.device = device
    
    def load_predictions(self, json_path: str) -> List[Dict[str, Any]]:
        """Load predictions from JSON file."""
        with open(json_path, "r", encoding="utf-8") as f:
            return json.load(f)
    
    def evaluate(self, json_path: str) -> Dict[str, Any]:
        """
        Evaluate a single prediction file.
        
        Metrics:
        - Accuracy: Computed on answers (exact match)
        - NLG metrics (BLEU, METEOR, ROUGE, CIDEr, BERTScore): Computed on explanations
        
        Args:
            json_path: Path to predictions JSON file
            
        Returns:
            Dictionary containing:
            - accuracy: Answer accuracy (exact match)
            - total_examples: Total number of examples
            - correct_count: Number of correct answers
            - unfiltered_scores: NLG metrics for explanations
            - by_answer_type: Metrics grouped by answer type
        """
        data = self.load_predictions(json_path)
        
        # Initialize containers
        all_pred_answers = []
        all_gt_answers = []
        all_gt_expls = []
        all_pred_expls = []
        by_type = {}
        
        # Process each example
        for item in data:
            # Extract answers (for accuracy)
            gt_ans = item.get("ground_truth", "")
            pred_ans = item.get("prediction", "")
            all_gt_answers.append(gt_ans)
            all_pred_answers.append(pred_ans)
            
            # Extract explanations (for NLG metrics)
            gt_expls_raw = item.get("gt_explanation", [])
            gt_expls = [normalize_explanation(e) for e in ensure_list(gt_expls_raw)]
            if not gt_expls or all(not e.strip() for e in gt_expls):
                gt_expls = [""]  # Fallback to empty string if no valid explanation
            
            pred_expl_raw = item.get("explanation", "")
            pred_expl = normalize_explanation(pred_expl_raw)
            
            all_gt_expls.append(gt_expls)
            all_pred_expls.append(pred_expl)
            
            # Group by answer type
            ans_type = item.get("answer_type", "other")
            if ans_type not in by_type:
                by_type[ans_type] = {
                    "gt_answers": [],
                    "pred_answers": [],
                    "gt_expls": [],
                    "pred_expls": []
                }
            
            by_type[ans_type]["gt_answers"].append(gt_ans)
            by_type[ans_type]["pred_answers"].append(pred_ans)
            by_type[ans_type]["gt_expls"].append(gt_expls)
            by_type[ans_type]["pred_expls"].append(pred_expl)
        
        # Compute overall metrics
        print("   📊 Computing answer accuracy...")
        accuracy, correct, total = compute_accuracy(all_pred_answers, all_gt_answers)
        print(f"   ✅ Answer accuracy: {accuracy:.2f}%")
        
        print("   📊 Computing explanation NLG metrics...")
        nlg_scores = compute_all_nlg_metrics(all_gt_expls, all_pred_expls, self.device)
        
        results = {
            "accuracy": accuracy,
            "total_examples": total,
            "correct_count": correct,
            "unfiltered_scores": nlg_scores,
            "by_answer_type": {}
        }
        
        # Compute metrics by answer type
        for ans_type, data_type in by_type.items():
            print(f"   📊 Computing metrics for answer_type: {ans_type}...")
            acc, corr, tot = compute_accuracy(data_type["pred_answers"], data_type["gt_answers"])
            scores = compute_all_nlg_metrics(data_type["gt_expls"], data_type["pred_expls"], self.device)
            
            results["by_answer_type"][ans_type] = {
                "accuracy": acc,
                "total_examples": tot,
                "correct_count": corr,
                "unfiltered_scores": scores
            }
        
        return results
