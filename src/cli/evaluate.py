import argparse
import json
from pathlib import Path

from vctp.core.config import load_experiment_config
from vctp.core.pipeline import VCTPPipeline
from vctp.core.registry import build_module
from vctp.core.utils import get_logger, process_answer
from vctp.data.loader import build_dataset


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    cfg = load_experiment_config(args.config).raw

    logger = get_logger("evaluate")
    see = build_module("see", cfg.get("see", {}).get("name", "noop-see"))  # type: ignore[arg-type]
    think = build_module("think", cfg.get("think", {}).get("name", "noop-think"))  # type: ignore[arg-type]
    confirm = build_module("confirm", cfg.get("confirm", {}).get("name", "noop-confirm"))  # type: ignore[arg-type]
    pipeline = VCTPPipeline(see=see, think=think, confirm=confirm)

    # Build dataset from dataset config if provided
    ds_cfg_path = cfg.get("datasets", {}).get(cfg.get("experiment", {}).get("dataset", ""), None)
    results = []
    if ds_cfg_path and isinstance(ds_cfg_path, str) and Path(ds_cfg_path).exists():
        ds_cfg = load_experiment_config(ds_cfg_path).raw
        dataset = build_dataset(ds_cfg, cfg.get("experiment", {}).get("split", "val"))
        for sample in dataset:
            row = pipeline.run(sample)
            row["answer_norm"] = process_answer(str(row.get("answer", "")))
            results.append(row)
    else:
        sample = {"image_path": "data/raw/placeholder.jpg", "question": "What is shown?"}
        row = pipeline.run(sample)
        row["answer_norm"] = process_answer(str(row.get("answer", "")))
        results.append(row)

    out_dir = Path(cfg.get("experiment", {}).get("output_dir", "runs/_placeholder"))
    out_dir.mkdir(parents=True, exist_ok=True)
    results_path = out_dir / "results.jsonl"
    with results_path.open("w", encoding="utf-8") as f:
        for r in results:
            f.write(json.dumps(r) + "\n")
    metrics = {
        "num_samples": len(results),
        "confirmed": sum(1 for r in results if r.get("confirmed")),
    }
    (out_dir / "metrics.json").write_text(json.dumps(metrics), encoding="utf-8")
    (out_dir / "config_resolved.yaml").write_text(json.dumps(cfg), encoding="utf-8")
    logger.info("Wrote results to %s", results_path)


if __name__ == "__main__":
    main()
