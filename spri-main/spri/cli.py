import argparse
import json
from .config import load_config
from .artifacts import write_json


def main(argv=None):
    parser = argparse.ArgumentParser(description="SPRI staged experiment pipeline")
    parser.add_argument("stage", choices=["prepare", "features", "train-din", "train-index", "materialize",
                                         "train", "evaluate", "report", "all", "config", "aggregate"])
    parser.add_argument("--config", action="append", default=[], help="YAML overlay; repeat in priority order")
    parser.add_argument("--set", action="append", default=[], dest="overrides", help="Strict dotted.key=value override")
    parser.add_argument("--resume", action="store_true", help="Resume an incomplete training stage from its last epoch")
    parser.add_argument("--results", nargs="+", help="Result JSON paths for aggregate")
    parser.add_argument("--output", help="Aggregate output JSON")
    args = parser.parse_args(argv)
    if args.stage == "aggregate":
        from .evaluation.aggregate import aggregate
        result = aggregate(args.results or [])
        if args.output:
            write_json(args.output, result)
        print(json.dumps(result, indent=2))
        return
    config = load_config(args.config, args.overrides)
    if args.stage == "config":
        print(json.dumps(config, indent=2))
        return
    from .pipeline import Pipeline
    Pipeline(config, resume=args.resume).run(args.stage)


if __name__ == "__main__":
    main()

