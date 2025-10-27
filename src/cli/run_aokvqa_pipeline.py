"""
Full Pipeline for AOKVQA using Refactored VCTP Framework
Based on VisualCoT/main_aokvqa.py but using the new modular architecture

Usage:
    python run_aokvqa_pipeline.py --config configs/experiments/aokvqa_baseline.yaml
    python run_aokvqa_pipeline.py --config configs/experiments/aokvqa_full.yaml
"""

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional
import yaml

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from vctp.core.pipeline import VCTPPipeline
from vctp.see.perception import VisualCoTPerception, NoOpPerception, CLIPOnlyPerception
from vctp.think.reasoner import VisualCoTReasoner, NoOpReasoner
from vctp.confirm.confirmer import (
    VisualConsistencyConfirmer,
    AnswerConsistencyConfirmer,
    NoOpConfirmer,
)
from vctp.think.llm.factory import create_llm_adapter
from vctp.think.prompts import FewShotExamplesManager
from vctp.data.loader import build_dataset


class AOKVQAPipeline:
    """Complete AOKVQA pipeline with all components."""

    def __init__(self, config_path: str):
        """
        Initialize pipeline from config.

        Args:
            config_path: Path to YAML configuration file
        """
        self.config = self._load_config(config_path)
        self.device = self._get_device()

        # Initialize components
        self.see_module = self._build_see_module()
        self.think_module = self._build_think_module()
        self.confirm_module = self._build_confirm_module()

        # Create pipeline
        self.pipeline = VCTPPipeline(
            see=self.see_module, think=self.think_module, confirm=self.confirm_module
        )

        # Load dataset
        self.dataset = self._build_dataset()

        print(f"Pipeline initialized with:")
        print(f"  SEE: {self.config['see']['name']}")
        print(f"  THINK: {self.config['think']['name']}")
        print(f"  CONFIRM: {self.config['confirm']['name']}")

    def _load_config(self, config_path: str) -> Dict:
        """Load configuration from YAML file."""
        with open(config_path, "r") as f:
            config = yaml.safe_load(f)

        # Load dataset config if referenced
        if "datasets" in config:
            dataset_name = config["experiment"]["dataset"]
            if dataset_name in config["datasets"]:
                dataset_config_path = config["datasets"][dataset_name]
                with open(dataset_config_path, "r") as f:
                    dataset_config = yaml.safe_load(f)
                config["dataset"] = dataset_config.get("dataset", dataset_config)

        return config

    def _get_device(self) -> str:
        """Get device for models."""
        import torch

        if torch.cuda.is_available():
            return "cuda"
        return "cpu"

    def _build_see_module(self):
        """Build SEE (perception) module from config."""
        see_config = self.config["see"]
        see_name = see_config["name"]

        if see_name == "noop-see":
            return NoOpPerception()

        elif see_name == "clip-only-perception":
            return CLIPOnlyPerception(
                model_name=see_config.get("clip_model", "openai/clip-vit-base-patch16"),
                device=self.device,
            )

        elif see_name == "visualcot-perception":
            dataset_cfg = self.config.get("dataset", {})

            return VisualCoTPerception(
                # Scene graph config
                sg_dir=dataset_cfg.get("scene_graph_dir"),
                sg_attr_dir=dataset_cfg.get("scene_graph_attr_dir"),
                sg_caption_dir=dataset_cfg.get("concept_caption_dir"),
                # Strategy
                iterative_strategy=see_config.get("iterative_strategy", "caption"),
                caption_type=see_config.get("caption_type", "vinvl"),
                # BLIP2 (optional)
                use_blip2=see_config.get("use_blip2", False),
                blip2_api_urls=see_config.get("blip2_api_urls"),
                # CLIP
                use_clip_features=see_config.get("use_clip", True),
                clip_model_name=see_config.get("clip_model", "openai/clip-vit-base-patch16"),
                device=self.device,
                debug=see_config.get("debug", False),
            )

        else:
            raise ValueError(f"Unknown SEE module: {see_name}")

    def _build_think_module(self):
        """Build THINK (reasoning) module from config."""
        think_config = self.config["think"]
        think_name = think_config["name"]

        if think_name == "noop-think":
            return NoOpReasoner()

        elif think_name == "visualcot-reasoner":
            # Create LLM adapter
            llm = self._create_llm(think_config)

            # Create few-shot examples manager
            examples_manager = self._create_examples_manager(think_config)

            return VisualCoTReasoner(
                llm=llm,
                examples_manager=examples_manager,
                engine=think_config.get("engine", "gpt3"),
                chain_of_thoughts=think_config.get("chain_of_thoughts", True),
                choice_only=think_config.get("choice_only", False),
                n_ensemble=think_config.get("n_ensemble", 1),
                use_thought_verification=think_config.get("use_thought_verification", False),
                debug=think_config.get("debug", False),
            )

        else:
            raise ValueError(f"Unknown THINK module: {think_name}")

    def _create_llm(self, think_config: Dict):
        """Create LLM adapter."""
        # Get API keys
        api_keys = []
        api_key_file = think_config.get("api_key_file")
        if api_key_file and os.path.exists(api_key_file):
            with open(api_key_file, "r") as f:
                api_keys = [line.strip() for line in f if line.strip()]

        # Or from environment
        import os

        if not api_keys:
            openai_key = os.getenv("OPENAI_API_KEY")
            if openai_key:
                api_keys = [openai_key]

        return create_llm_adapter(
            engine=think_config.get("engine", "gpt3"),
            engine_name=think_config.get("engine_name"),
            api_keys=api_keys,
            temperature=think_config.get("temperature", 0.0),
            max_tokens=think_config.get("max_tokens", 41),
            device=self.device,
            debug=think_config.get("debug", False),
        )

    def _create_examples_manager(self, think_config: Dict):
        """Create few-shot examples manager."""
        # This would load from preprocessed data
        # For now, create empty manager
        from vctp.think.prompts import FewShotExamplesManager

        examples_file = think_config.get("examples_file")
        if examples_file and os.path.exists(examples_file):
            with open(examples_file, "r") as f:
                examples_data = json.load(f)
        else:
            examples_data = {}

        return FewShotExamplesManager(examples_data)

    def _build_confirm_module(self):
        """Build CONFIRM (verification) module from config."""
        confirm_config = self.config["confirm"]
        confirm_name = confirm_config["name"]

        if confirm_name == "noop-confirm":
            return NoOpConfirmer()

        elif confirm_name == "visual-consistency":
            return VisualConsistencyConfirmer(
                method=confirm_config.get("method", "clip"),
                verify_threshold=confirm_config.get("threshold", 0.0),
                device=self.device,
                debug=confirm_config.get("debug", False),
            )

        elif confirm_name == "answer-consistency":
            return AnswerConsistencyConfirmer(
                correct_answer=confirm_config.get("correct_answer", True),
                device=self.device,
                debug=confirm_config.get("debug", False),
            )

        else:
            raise ValueError(f"Unknown CONFIRM module: {confirm_name}")

    def _build_dataset(self):
        """Build dataset iterator."""
        dataset_cfg = self.config.get("dataset", {})
        split = self.config["experiment"].get("split", "val")

        return build_dataset(dataset_cfg, split)

    def run(self, limit: Optional[int] = None, save_results: bool = True):
        """
        Run pipeline on dataset.

        Args:
            limit: Limit number of samples (None = all)
            save_results: Whether to save results to file
        """
        results = []

        print(f"\nRunning pipeline on {self.config['experiment']['dataset']} dataset...")
        print(f"Split: {self.config['experiment']['split']}")

        for idx, sample in enumerate(self.dataset):
            if limit and idx >= limit:
                break

            print(f"\n[{idx+1}] Processing sample {sample['question_id']}...")
            print(f"  Question: {sample['question'][:80]}...")

            try:
                result = self.pipeline.run(sample)
                results.append(result)

                print(f"  Answer: {result['answer']}")
                print(f"  Confirmed: {result['confirmed']} (score: {result['score']:.3f})")

            except Exception as e:
                print(f"  ERROR: {e}")
                import traceback

                traceback.print_exc()

        # Save results
        if save_results:
            self._save_results(results)

        # Print summary
        self._print_summary(results)

        return results

    def _save_results(self, results: List[Dict]):
        """Save results to output directory."""
        output_dir = self.config["experiment"].get("output_dir", "runs/default")
        os.makedirs(output_dir, exist_ok=True)

        output_file = os.path.join(output_dir, "results.json")

        with open(output_file, "w") as f:
            json.dump(results, f, indent=2)

        print(f"\nResults saved to: {output_file}")

    def _print_summary(self, results: List[Dict]):
        """Print summary statistics."""
        if not results:
            print("\nNo results to summarize.")
            return

        total = len(results)
        confirmed = sum(1 for r in results if r.get("confirmed", False))
        avg_score = sum(r.get("score", 0.0) for r in results) / total if total > 0 else 0.0

        print(f"\n{'='*70}")
        print(f"SUMMARY")
        print(f"{'='*70}")
        print(f"  Total samples: {total}")
        print(f"  Confirmed: {confirmed} ({confirmed/total*100:.1f}%)")
        print(f"  Average score: {avg_score:.3f}")
        print(f"{'='*70}")


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Run AOKVQA pipeline with refactored VCTP")
    parser.add_argument(
        "--config",
        type=str,
        default="configs/experiments/aokvqa_baseline.yaml",
        help="Path to config file",
    )
    parser.add_argument(
        "--limit", type=int, default=None, help="Limit number of samples to process"
    )
    parser.add_argument("--no-save", action="store_true", help="Do not save results")

    args = parser.parse_args()

    # Create and run pipeline
    pipeline = AOKVQAPipeline(args.config)
    pipeline.run(limit=args.limit, save_results=not args.no_save)


if __name__ == "__main__":
    main()
