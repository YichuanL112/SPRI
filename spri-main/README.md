# SPRI

**Shared-Prefix Refinement Indexing for Generative Recommendation**

SPRI combines prefix-guided quantization (PGQ), prefix-conditioned behavioral refinement (PCR), and structured history construction (SHC) for generative recommendation. This repository implements the pipeline from data preparation and feature extraction to two-stage index learning, T5 training, and evaluation on Beauty, Sports, Toys, and Yelp.

## Installation

Use Python 3.10 or newer with a PyTorch build suitable for your hardware. Run the following commands from this directory:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Quick Start

Provide Amazon Reviews 2014 review and metadata files, or Yelp review and business files. Yelp inputs are assumed to match the dataset definition in the paper. Data and pretrained weights are not bundled; Sentence-T5 weights are downloaded on first use.

Run the full pipeline on Beauty:

```bash
python main.py all \
  --config configs/default.yaml \
  --config configs/datasets/beauty.yaml \
  --set data.reviews=/path/to/reviews_Beauty_5.json.gz \
  --set data.metadata=/path/to/meta_Beauty.json.gz \
  --set runtime.run_id=beauty-main \
  --set runtime.seed=2025
```

For other datasets, use `sports.yaml`, `toys.yaml`, or `yelp.yaml` and update the input paths. Shared settings are in `configs/default.yaml`; dataset-specific settings are in `configs/datasets/`. Use `--set key=value` to override settings. Relative paths are resolved from the working directory.

The pipeline runs these stages in order:

```text
prepare → features → train-din → train-index → materialize → train → evaluate → report
```

To run an individual stage, replace `all` with its name and keep the same configuration arguments; prerequisite stages must already be complete. Add `--resume` to continue interrupted training. Replace `all` with `config` to inspect settings, or run `python main.py --help` for available options.

Checkpoints, the item index, predictions, and metrics are saved under `runs/<dataset>/<run_id>/seed-<seed>/`. Final test metrics are in `test_result.json`.
