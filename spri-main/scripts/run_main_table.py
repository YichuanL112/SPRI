"""Run explicitly configured seeds. Outputs are newly measured, not paper reference values."""
import argparse
import subprocess
import sys


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", action="append", required=True)
    parser.add_argument("--seeds", nargs="+", type=int, default=[2024, 2025, 2026, 2027, 2028])
    parser.add_argument("--run-id", default="main")
    args = parser.parse_args()
    for seed in args.seeds:
        cmd = [sys.executable, "main.py", "all"]
        for config in args.config:
            cmd += ["--config", config]
        cmd += ["--set", f"runtime.seed={seed}", "--set", f"runtime.run_id={args.run_id}", "--resume"]
        subprocess.run(cmd, check=True)


if __name__ == "__main__":
    main()

