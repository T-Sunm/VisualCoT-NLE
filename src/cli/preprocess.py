import argparse
import json
from pathlib import Path

from vctp.core.config import load_experiment_config
from vctp.core.utils import get_logger
from vctp.data.loader import build_dataset


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    cfg = load_experiment_config(args.config).raw

    logger = get_logger("preprocess")
    ds_cfg_path = cfg.get("datasets", {}).get(cfg.get("experiment", {}).get("dataset", ""), None)
    if not ds_cfg_path:
        out = Path("data/processed/_placeholder")
        out.mkdir(parents=True, exist_ok=True)
        (out / "ok.txt").write_text("preprocess placeholder\n", encoding="utf-8")
        logger.info("No dataset config found; wrote placeholder preprocess outputs to %s", out)
        return

    ds_cfg = load_experiment_config(ds_cfg_path).raw
    dataset_name = (ds_cfg.get("dataset") or {}).get("name", "dataset")
    dataset = build_dataset(ds_cfg, cfg.get("experiment", {}).get("split", "val"))
    out = Path(f"data/processed/{dataset_name}")
    out.mkdir(parents=True, exist_ok=True)
    meta_path = out / "metadata.jsonl"
    with meta_path.open("w", encoding="utf-8") as f:
        for sample in dataset:
            f.write(json.dumps(sample) + "\n")
    logger.info("Wrote metadata to %s", meta_path)


if __name__ == "__main__":
    main()
