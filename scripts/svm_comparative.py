"""
Comparative SVM Training on All Available Datasets
Dynamically discovers and trains SVM on all datasets defined in preprocessing.py
When new datasets are added to dataset_config, they're automatically included
"""

import os
import sys
sys.path.insert(0, os.path.dirname(__file__))

from svm_classifier import SVMClassifierWithCV
from preprocessing import GenomicDataProcessor
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def discover_available_datasets():
    """
    Dynamically discover all available datasets from preprocessing.py

    Returns:
        dict: Dictionary of dataset_name: dataset_config
    """
    processor = GenomicDataProcessor()
    datasets = processor.dataset_config

    logger.info(f"Discovered {len(datasets)} available datasets:")
    for dataset_name, config in datasets.items():
        logger.info(f"  - {dataset_name}: {config['n_cancer']} cancer + {config['n_normal']} normal samples")

    return datasets


def get_dataset_info(dataset_name, config):
    """
    Generate dataset info from config

    Args:
        dataset_name (str): Dataset ID (e.g., "GSE19804")
        config (dict): Dataset configuration from dataset_config

    Returns:
        dict: Dataset info with name, description, folder
    """
    total_samples = config['n_cancer'] + config['n_normal']
    ratio = f"{config['n_cancer']}/{config['n_normal']}"

    # Determine if balanced or imbalanced
    if config['n_cancer'] == config['n_normal']:
        balance_str = "balanced"
    else:
        balance_str = "imbalanced"

    # Try to identify cancer type from dataset name
    cancer_type = "Unknown"
    if "19804" in dataset_name:
        cancer_type = "Lung Cancer"
    elif "42568" in dataset_name:
        cancer_type = "Breast Cancer"

    return {
        "name": f"{cancer_type}",
        "description": f"{dataset_name} - {cancer_type} ({balance_str}: {ratio}, {total_samples} total)",
        "folder": f"{dataset_name}_{cancer_type.lower().replace(' ', '_')}",
        "config": config
    }


def print_header(num_datasets):
    """Print project header"""
    print("\n" + "="*80)
    print("COMPREHENSIVE SVM CLASSIFIER ANALYSIS")
    print(f"Training on {num_datasets} Available Gene Expression Datasets")
    print("="*80)
    print("\nResults will be organized in dataset-specific subdirectories:")
    print("  results/<DATASET_ID>_<CANCER_TYPE>/")
    print("="*80 + "\n")


def main():
    """Run SVM on all available datasets"""

    # Discover available datasets
    datasets = discover_available_datasets()

    print_header(len(datasets))

    results_summary = {}
    completed_count = 0
    failed_count = 0

    # Run SVM on each dataset
    for dataset_name, config in datasets.items():
        dataset_info = get_dataset_info(dataset_name, config)

        logger.info("\n" + "="*80)
        logger.info(f"DATASET: {dataset_info['name']}")
        logger.info(f"ID: {dataset_name}")
        logger.info(f"Description: {dataset_info['description']}")
        logger.info("="*80)

        try:
            classifier = SVMClassifierWithCV(
                dataset_name=dataset_name,
                n_splits=5,
                random_state=42
            )
            classifier.run_full_pipeline()

            # Store results info
            results_summary[dataset_name] = {
                'status': 'COMPLETED',
                'results_dir': classifier.results_dir,
                'description': dataset_info['description'],
                'config': config
            }

            completed_count += 1

            logger.info(f"\n✓ {dataset_name} COMPLETED SUCCESSFULLY")
            logger.info(f"✓ Results saved to: {classifier.results_dir}\n")

        except Exception as e:
            logger.error(f"✗ {dataset_name} FAILED: {e}\n")
            results_summary[dataset_name] = {
                'status': 'FAILED',
                'error': str(e),
                'description': dataset_info['description'],
                'config': config
            }
            failed_count += 1
            continue

    # Print final summary
    print("\n" + "="*80)
    print("COMPREHENSIVE ANALYSIS COMPLETE")
    print("="*80)
    print(f"\nResults Summary:")
    print(f"  ✓ Completed: {completed_count}/{len(datasets)}")
    print(f"  ✗ Failed: {failed_count}/{len(datasets)}")

    print("\nDataset-by-Dataset Results:\n")

    for dataset_name, info in results_summary.items():
        print(f"{dataset_name}")
        print(f"  Cancer Type: {info['description'].split(' - ')[1].split('(')[0].strip()}")
        print(f"  Samples: {info['config']['n_cancer']} cancer + {info['config']['n_normal']} normal")
        print(f"  Status: {info['status']}")
        if info['status'] == 'COMPLETED':
            print(f"  Results: {info['results_dir']}")
        else:
            print(f"  Error: {info.get('error', 'Unknown')}")
        print()

    print("="*80)
    print("To view results, check:")
    for dataset_name, info in results_summary.items():
        if info['status'] == 'COMPLETED':
            print(f"  • {info['results_dir']}")
    print("="*80 + "\n")


if __name__ == "__main__":
    main()


