import argparse
import json
from pathlib import Path

from vctp.core.config import load_experiment_config
from vctp.core.pipeline import VCTPPipeline
from vctp.core.registry import build_module
from vctp.core.utils import get_logger


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--image")
    group.add_argument("--input_jsonl")
    parser.add_argument("--question")
    args = parser.parse_args()

    cfg = load_experiment_config(args.config).raw
    logger = get_logger("inference")
    see = build_module("see", cfg.get("see", {}).get("name", "noop-see"))  # type: ignore[arg-type]
    think = build_module("think", cfg.get("think", {}).get("name", "noop-think"))  # type: ignore[arg-type]
    confirm = build_module("confirm", cfg.get("confirm", {}).get("name", "noop-confirm"))  # type: ignore[arg-type]
    pipeline = VCTPPipeline(see=see, think=think, confirm=confirm)

    samples = []
    if args.image:
        if not args.question:
            raise SystemExit("--question is required when using --image")
        samples = [{"image_path": args.image, "question": args.question}]
    else:
        for line in Path(args.input_jsonl).read_text(encoding="utf-8").splitlines():
            samples.append(json.loads(line))

    for s in samples:
        res = pipeline.run(s)
        logger.info(json.dumps(res))


if __name__ == "__main__":
    main()
