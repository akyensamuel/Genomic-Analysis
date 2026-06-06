"""
Comparative SVM Training on All Available Datasets
====================================================
Dynamically discovers every dataset registered in GenomicDataProcessor.DATASET_CONFIG
and runs the full SVM pipeline (Path A baseline + all Path B feature-selection methods)
on each one.

Usage
-----
    python svm_comparative.py [--n-splits N] [--random-state S]

Fix log (vs original)
---------------------
- logging.basicConfig moved inside main() so importing this module never
  reconfigures the root logger.  The module-level basicConfig in the
  original conflicted with svm_classifier._setup_logging().
- discover_available_datasets() now reads from GenomicDataProcessor.DATASET_CONFIG
  (the class-level dict) instead of instantiating a throw-away processor
  object just to read metadata.  This removes the cwd() dependency for
  dataset discovery.
- Added argparse so n_splits and random_state can be changed from the
  command line without editing the source.
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

current_dir = Path(__file__).resolve().parent
if str(current_dir) not in sys.path:
    sys.path.insert(0, str(current_dir))

from preprocessing import GenomicDataProcessor
from svm_classifier import SVMClassifierWithCV

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Dataset discovery
# ---------------------------------------------------------------------------
def discover_available_datasets() -> dict:
    """
    Read the dataset registry directly from the class attribute — no
    filesystem access or cwd() dependency.
    """
    datasets = GenomicDataProcessor.DATASET_CONFIG

    logger.info(f"Discovered {len(datasets)} dataset(s):")
    for name, cfg in datasets.items():
        cancer_type = cfg.get("cancer_type", "Unknown")
        logger.info(
            f"  {name} ({cancer_type}): "
            f"{cfg['n_cancer']} cancer + {cfg['n_normal']} normal"
        )

    return datasets


def get_dataset_meta(dataset_name: str, config: dict) -> dict:
    """Build display metadata from a single dataset config entry."""
    total = config["n_cancer"] + config["n_normal"]
    ratio = f"{config['n_cancer']}/{config['n_normal']}"
    balance = "balanced" if config["n_cancer"] == config["n_normal"] else "imbalanced"
    cancer_type = config.get("cancer_type", "Unknown Cancer")

    return {
        "cancer_type": cancer_type,
        "description": f"{dataset_name} - {cancer_type} ({balance}: {ratio}, {total} total)",
        "folder": f"{dataset_name}_{cancer_type.lower().replace(' ', '_')}",
        "config": config,
    }


# ---------------------------------------------------------------------------
# Display helpers
# ---------------------------------------------------------------------------
def print_header(num_datasets: int) -> None:
    print("\n" + "=" * 80)
    print("COMPREHENSIVE SVM CLASSIFIER ANALYSIS")
    print(f"Training on {num_datasets} available gene expression dataset(s)")
    print("=" * 80)
    print("\nResults organised in dataset-specific subdirectories:")
    print("  results/<DATASET_ID>_<CANCER_TYPE>/")
    print("=" * 80 + "\n")


def print_final_summary(results_summary: dict) -> None:
    completed = sum(1 for v in results_summary.values() if v["status"] == "COMPLETED")
    failed    = len(results_summary) - completed
    total     = len(results_summary)

    print("\n" + "=" * 80)
    print("COMPREHENSIVE ANALYSIS COMPLETE")
    print("=" * 80)
    print(f"\nResults summary:")
    print(f"  ✓ Completed : {completed}/{total}")
    print(f"  ✗ Failed    : {failed}/{total}")
    print("\nDataset-by-dataset:\n")

    for name, info in results_summary.items():
        cfg = info["config"]
        print(f"{name}")
        print(f"  Cancer type : {info['cancer_type']}")
        print(f"  Samples     : {cfg['n_cancer']} cancer + {cfg['n_normal']} normal")
        print(f"  Status      : {info['status']}")
        if info["status"] == "COMPLETED":
            print(f"  Results dir : {info['results_dir']}")
        else:
            print(f"  Error       : {info.get('error', 'unknown')}")
        print()

    print("=" * 80)
    print("Output locations:")
    for info in results_summary.values():
        if info["status"] == "COMPLETED":
            print(f"  • {info['results_dir']}")
    print("=" * 80 + "\n")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main(n_splits: int = 5, random_state: int = 42) -> None:
    datasets = discover_available_datasets()
    print_header(len(datasets))

    results_summary: dict = {}

    for dataset_name, config in datasets.items():
        meta = get_dataset_meta(dataset_name, config)

        logger.info("\n" + "=" * 80)
        logger.info(f"DATASET : {meta['cancer_type']}")
        logger.info(f"ID      : {dataset_name}")
        logger.info(f"Desc    : {meta['description']}")
        logger.info("=" * 80)

        try:
            classifier = SVMClassifierWithCV(
                dataset_name=dataset_name,
                n_splits=n_splits,
                random_state=random_state,
            )
            classifier.run_full_pipeline()

            results_summary[dataset_name] = {
                "status":      "COMPLETED",
                "results_dir": classifier.results_dir,
                "cancer_type": meta["cancer_type"],
                "config":      config,
            }
            logger.info(f"\n✓ {dataset_name} completed — results at: {classifier.results_dir}\n")

        except Exception as exc:
            logger.error(f"✗ {dataset_name} FAILED: {exc}\n")
            results_summary[dataset_name] = {
                "status":      "FAILED",
                "error":       str(exc),
                "cancer_type": meta["cancer_type"],
                "config":      config,
            }

    print_final_summary(results_summary)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run the full SVM pipeline on every registered dataset.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--n-splits", type=int, default=5, dest="n_splits",
        help="Number of stratified cross-validation folds",
    )
    parser.add_argument(
        "--random-state", type=int, default=42, dest="random_state",
        help="Random seed for reproducibility",
    )
    return parser


if __name__ == "__main__":
    # Configure logging here — not at module level — so importing this file
    # never reconfigures the root logger.
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s  %(levelname)-8s  %(message)s",
        datefmt="%H:%M:%S",
    )

    args = _build_parser().parse_args()
    main(n_splits=args.n_splits, random_state=args.random_state)
