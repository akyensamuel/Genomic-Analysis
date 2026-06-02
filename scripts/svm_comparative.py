"""
Comparative SVM Training on All Available Datasets
Dynamically discovers and trains SVM on all datasets defined in preprocessing.py
"""

import logging
import sys
from pathlib import Path

# Modern, robust path handling
current_dir = Path(__file__).resolve().parent
if str(current_dir) not in sys.path:
    sys.path.insert(0, str(current_dir))

from preprocessing import GenomicDataProcessor
from svm_classifier import SVMClassifierWithCV

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def discover_available_datasets():
    """
    Dynamically discover all available datasets from preprocessing.py
    """
    processor = GenomicDataProcessor()
    datasets = processor.dataset_config

    logger.info(f"Discovered {len(datasets)} available datasets:")
    for dataset_name, config in datasets.items():
        cancer_type = config.get("cancer_type", "Unknown Cancer")
        logger.info(
            f"  - {dataset_name} ({cancer_type}): "
            f"{config['n_cancer']} cancer + {config['n_normal']} normal samples"
        )

    return datasets


def get_dataset_info(dataset_name, config):
    """
    Generate clean dataset metadata directly from the configuration dictionary.
    """
    total_samples = config["n_cancer"] + config["n_normal"]
    ratio = f"{config['n_cancer']}/{config['n_normal']}"
    balance_str = "balanced" if config["n_cancer"] == config["n_normal"] else "imbalanced"
    
    # Read directly from config with a fallback safety net
    cancer_type = config.get("cancer_type", "Unknown Cancer")

    return {
        "cancer_type": cancer_type,
        "description": f"{dataset_name} - {cancer_type} ({balance_str}: {ratio}, {total_samples} total)",
        "folder": f"{dataset_name}_{cancer_type.lower().replace(' ', '_')}",
        "config": config,
    }


def print_header(num_datasets):
    """Print clean terminal header"""
    print("\n" + "=" * 80)
    print("COMPREHENSIVE SVM CLASSIFIER ANALYSIS")
    print(f"Training on {num_datasets} Available Gene Expression Datasets")
    print("=" * 80)
    print("\nResults organized in dataset-specific subdirectories:")
    print("  results/<DATASET_ID>_<CANCER_TYPE>/")
    print("=" * 80 + "\n")


def main(n_splits=5, random_state=42):
    """Run SVM on all available datasets with configurable hyperparameters"""
    datasets = discover_available_datasets()
    print_header(len(datasets))

    results_summary = {}
    completed_count = 0
    failed_count = 0

    for dataset_name, config in datasets.items():
        meta = get_dataset_info(dataset_name, config)

        logger.info("\n" + "=" * 80)
        logger.info(f"DATASET: {meta['cancer_type']}")
        logger.info(f"ID: {dataset_name}")
        logger.info(f"Description: {meta['description']}")
        logger.info("=" * 80)

        try:
            # Instantiating the pipeline using our passed config defaults
            classifier = SVMClassifierWithCV(
                dataset_name=dataset_name,
                n_splits=n_splits,
                random_state=random_state,
            )
            classifier.run_full_pipeline()

            results_summary[dataset_name] = {
                "status": "COMPLETED",
                "results_dir": classifier.results_dir,
                "cancer_type": meta["cancer_type"],
                "config": config,
            }
            completed_count += 1
            logger.info(f"\n✓ {dataset_name} COMPLETED SUCCESSFULLY")
            logger.info(f"✓ Results saved to: {classifier.results_dir}\n")

        except Exception as e:
            logger.error(f"✗ {dataset_name} FAILED: {e}\n")
            results_summary[dataset_name] = {
                "status": "FAILED",
                "error": str(e),
                "cancer_type": meta["cancer_type"],
                "config": config,
            }
            failed_count += 1

    # Final Summary Report
    print("\n" + "=" * 80)
    print("COMPREHENSIVE ANALYSIS COMPLETE")
    print("=" * 80)
    print(f"\nResults Summary:")
    print(f"  ✓ Completed: {completed_count}/{len(datasets)}")
    print(f"  ✗ Failed:    {failed_count}/{len(datasets)}")
    print("\nDataset-by-Dataset Results:\n")

    for dataset_name, info in results_summary.items():
        print(f"{dataset_name}")
        print(f"  Cancer Type: {info['cancer_type']}")
        print(f"  Samples:     {info['config']['n_cancer']} cancer + {info['config']['n_normal']} normal")
        print(f"  Status:      {info['status']}")
        if info["status"] == "COMPLETED":
            print(f"  Results:     {info['results_dir']}")
        else:
            print(f"  Error:       {info.get('error', 'Unknown Error')}")
        print()

    print("=" * 80)
    print("To view results, check:")
    for dataset_name, info in results_summary.items():
        if info["status"] == "COMPLETED":
            print(f"  • {info['results_dir']}")
    print("=" * 80 + "\n")


if __name__ == "__main__":
    # Allows you to easily change cross-validation constraints globally
    main(n_splits=5, random_state=42)