"""
Main evaluation script for VQA predictions.
Usage: python evaluate.py --input-dir results --device cuda
"""

import os
import argparse
from datetime import datetime
import pandas as pd
from evaluator import VQAEvaluator


# Configuration: List of files to evaluate (empty = evaluate all JSON files in input_dir)
FILES_TO_EVALUATE = ['250_curr_anstype.json']


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
    
    # By answer_type rows
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
    args = parser.parse_args()
    
    # Determine files to evaluate
    if FILES_TO_EVALUATE:
        files = [f if f.endswith(".json") else f"{f}.json" for f in FILES_TO_EVALUATE]
    else:
        files = sorted([f for f in os.listdir(args.input_dir) 
                       if f.endswith(".json") and "_score" not in f and "summary" not in f])
    
    print(f"📁 Evaluating {len(files)} file(s) from {args.input_dir}")
    
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
