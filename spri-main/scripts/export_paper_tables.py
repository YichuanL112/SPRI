"""Render an aggregate JSON as CSV; missing statistics stay empty."""
import argparse
import csv
import json
import sys


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("aggregate")
    args = parser.parse_args()
    with open(args.aggregate, encoding="utf-8") as f:
        report = json.load(f)
    writer = csv.writer(sys.stdout)
    writer.writerow(["dataset", "kind", "runs", "metric", "mean", "sample_std"])
    for metric, values in report["metrics"].items():
        writer.writerow([report["dataset"], report["experiment_kind"], report["n_runs"],
                         metric, values["mean"], values["std"]])

