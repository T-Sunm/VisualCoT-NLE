import os
import sys
import json
import argparse
from datetime import datetime
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from src.evaluation.evaluator import VQAEvaluator



# Configuration: List of files to evaluate (empty = evaluate all JSON files in input_dir)
FILES_TO_EVALUATE = ['visual_cot_results.json']

# Path to original annotations
ANNOTATIONS_PATH = "/home/research/workspace/data/raw/vivqa-x/annotations/test.json"

def enrich_predictions_with_answer_type(predictions_path: str, annotations_path: str) -> None:
    """
    Add answer_type and ground truth explanations from annotations to prediction file.
    
    Args:
        predictions_path: Path to inference results JSON
        annotations_path: Path to original annotations JSON
    """
    # Load both files
    with open(annotations_path, 'r', encoding='utf-8') as f:
        annotations = json.load(f)
    
    with open(predictions_path, 'r', encoding='utf-8') as f:
        predictions = json.load(f)
    
    predictions = predictions[:300]
    # Create question_id to annotation mapping
    ann_map = {ann['question_id']: ann for ann in annotations}
    
    # Add answer_type and ground truth explanations to predictions
    enriched_count = 0
    for pred in predictions:
        qid = pred.get('question_id')
        if qid in ann_map:
            ann = ann_map[qid]
            pred['answer_type'] = ann.get('answer_type', 'other')
            pred['gt_explanation'] = ann.get('explanation', [])
            enriched_count += 1
    
    # Save enriched predictions
    with open(predictions_path, 'w', encoding='utf-8') as f:
        json.dump(predictions, f, ensure_ascii=False, indent=2)
    
    print(f"   ✅ Enriched {enriched_count}/{len(predictions)} samples with answer_type and gt_explanation")



def format_results_as_dataframe(results: dict, model_name: str) -> pd.DataFrame:
    """
    Format evaluation results as a pandas DataFrame.
    
    Args:
        results: Evaluation results dictionary
        model_name: Name of the model
        
    Returns:
        DataFrame with evaluation metrics
    """
    rows = []
    
    # Overall row
    rows.append({
        "model": model_name,
        "answer_type": "Overall",
        "total": results["total_examples"],
        "correct": results["correct_count"],
        "accuracy": round(results["accuracy"], 2),
        **{k: round(v, 2) for k, v in results["unfiltered_scores"].items()}
    })
    
    # By answer_type rows - hiển thị tất cả
    for ans_type, type_data in results["by_answer_type"].items():
        rows.append({
            "model": model_name,
            "answer_type": ans_type,
            "total": type_data["total_examples"],
            "correct": type_data["correct_count"],
            "accuracy": round(type_data["accuracy"], 2),
            **{k: round(v, 2) for k, v in type_data["unfiltered_scores"].items()}
        })
    
    return pd.DataFrame(rows)



def main():
    """Main evaluation entry point."""
    parser = argparse.ArgumentParser(description="Evaluate VQA predictions")
    parser.add_argument("--input-dir", type=str, default="results", 
                       help="Directory containing prediction JSON files")
    parser.add_argument("--device", type=str, default="cuda", 
                       help="Device for BERTScore (cuda/cpu)")
    parser.add_argument("--annotations", type=str, default=ANNOTATIONS_PATH,
                       help="Path to original annotations file")
    parser.add_argument("--skip-enrich", action="store_true",
                       help="Skip enriching predictions with answer_type")
    args = parser.parse_args()
    
    # Determine files to evaluate
    if FILES_TO_EVALUATE:
        files = [f if f.endswith(".json") else f"{f}.json" for f in FILES_TO_EVALUATE]
    else:
        files = sorted([f for f in os.listdir(args.input_dir) 
                       if f.endswith(".json") and "_score" not in f and "summary" not in f])
    
    print(f"📁 Evaluating {len(files)} file(s) from {args.input_dir}")
    
    # Enrich predictions with answer_type
    if not args.skip_enrich and os.path.exists(args.annotations):
        print(f"\n🔧 Enriching predictions with answer_type from {args.annotations}")
        for fname in files:
            fpath = os.path.join(args.input_dir, fname)
            enrich_predictions_with_answer_type(fpath, args.annotations)
    
    # Initialize evaluator
    evaluator = VQAEvaluator(device=args.device)
    
    # Evaluate each file
    all_dfs = []
    for fname in files:
        fpath = os.path.join(args.input_dir, fname)
        print(f"\n🔎 Evaluating: {fname}")
        
        results = evaluator.evaluate(fpath)
        model_name = os.path.splitext(fname)[0]
        df = format_results_as_dataframe(results, model_name)
        all_dfs.append(df)
        
        print(f"   ✅ {model_name}: Accuracy={results['accuracy']:.2f}%")
    
    # Combine and save results
    final_df = pd.concat(all_dfs, ignore_index=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Use first model name for output filename
    first_model = os.path.splitext(files[0])[0]
    csv_path = os.path.join(args.input_dir, f"evaluate_{first_model}_{timestamp}.csv")
    
    final_df.to_csv(csv_path, index=False, encoding="utf-8")
    
    print(f"\n✅ Results saved to: {csv_path}")
    print(f"\n{final_df.to_string(index=False)}")



if __name__ == "__main__":
    main()