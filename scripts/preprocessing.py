"""
Unified Preprocessing Pipeline
Replicates Jupyter notebook logic in modular, reusable format
Handles data ingestion, transformation, and artifact storage
"""

import os
import numpy as np
import pandas as pd
from scipy import stats
import gzip
import urllib.request
import logging

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class GenomicDataProcessor:
    """Centralized preprocessing pipeline for gene expression data"""

    def __init__(self, dataset_name="GSE19804", project_dir=""):
        """
        Initialize processor with dataset selection

        Args:
            dataset_name (str): "GSE42568" or "GSE19804"
            project_dir (str): Project root directory
        """
        self.dataset_name = dataset_name
        self.project_dir = project_dir or os.getcwd()
        self.datasets_dir = os.path.join(self.project_dir, "datasets")
        self.preprocessed_dir = os.path.join(self.project_dir, "preprocessed_datasets")

        # Create directories if they don't exist
        os.makedirs(self.datasets_dir, exist_ok=True)
        os.makedirs(self.preprocessed_dir, exist_ok=True)

        self.dataset_config = {
            "GSE42568": {
                "url": "https://ftp.ncbi.nlm.nih.gov/geo/series/GSE42nnn/GSE42568/matrix/GSE42568_series_matrix.txt.gz",
                "filename": "GSE42568_series_matrix.txt.gz",
                "n_cancer": 104,
                "n_normal": 17,
            },
            "GSE19804": {
                "url": "https://ftp.ncbi.nlm.nih.gov/geo/series/GSE19nnn/GSE19804/matrix/GSE19804_series_matrix.txt.gz",
                "filename": "GSE19804_series_matrix.txt.gz",
                "n_cancer": 60,
                "n_normal": 60,
            }
        }

        self.raw_data = None
        self.X_raw = None
        self.X_log = None
        self.X_scaled = None
        self.y = None

    def download_dataset(self):
        """Automatically download dataset if not present"""
        config = self.dataset_config[self.dataset_name]
        filepath = os.path.join(self.datasets_dir, config["filename"])

        if os.path.exists(filepath):
            logger.info(f"Dataset already exists: {filepath}")
            return filepath

        logger.info(f"Downloading {self.dataset_name}...")
        try:
            urllib.request.urlretrieve(config["url"], filepath)
            logger.info(f"Downloaded successfully to {filepath}")
            return filepath
        except Exception as e:
            logger.error(f"Download failed: {e}")
            raise

    def load_data(self):
        """Load and structure the downloaded dataset"""
        logger.info(f"Loading {self.dataset_name}...")

        filepath = self.download_dataset()
        config = self.dataset_config[self.dataset_name]

        # Load, skipping metadata lines starting with '!'
        df = pd.read_csv(filepath, compression='infer', sep='\t',
                        comment='!', index_col=0)

        # Transpose: rows = samples, columns = genes
        self.X_raw = df.T

        # Create labels: 1 = Cancer, 0 = Normal
        self.y = np.array([1] * config["n_cancer"] + [0] * config["n_normal"])

        logger.info(f"Data loaded: {self.X_raw.shape[0]} samples, {self.X_raw.shape[1]} genes")
        logger.info(f"Class distribution: {config['n_cancer']} cancer, {config['n_normal']} normal")

        return self.X_raw, self.y

    def apply_log_transformation(self):
        """Apply Log2 transformation: log2(x + 1)"""
        logger.info("Applying Log2 transformation...")

        if self.X_raw is None:
            raise ValueError("Load data first using load_data()")

        self.X_log = np.log2(self.X_raw + 1)
        logger.info(f"Log transformation complete. Shape: {self.X_log.shape}")

        return self.X_log

    def apply_standardization(self):
        """Apply Z-score standardization: (x - μ) / σ"""
        logger.info("Applying Z-score standardization...")

        if self.X_log is None:
            raise ValueError("Apply log transformation first")

        from sklearn.preprocessing import StandardScaler

        scaler = StandardScaler()
        X_scaled_array = scaler.fit_transform(self.X_log)
        self.X_scaled = pd.DataFrame(
            X_scaled_array,
            index=self.X_log.index,
            columns=self.X_log.columns
        )

        logger.info(f"Standardization complete. Shape: {self.X_scaled.shape}")

        return self.X_scaled

    def preprocess_complete(self):
        """Execute full preprocessing pipeline"""
        logger.info("="*60)
        logger.info("STARTING PREPROCESSING PIPELINE")
        logger.info("="*60)

        self.load_data()
        self.apply_log_transformation()
        self.apply_standardization()

        logger.info("="*60)
        logger.info("PREPROCESSING COMPLETE")
        logger.info("="*60)

        return self.X_scaled, self.y

    def profile_data(self, stage="raw"):
        """Generate data profiling summary"""
        logger.info(f"\n--- DATA PROFILE ({stage.upper()}) ---")

        if stage == "raw":
            X = self.X_raw
        elif stage == "log":
            X = self.X_log
        elif stage == "scaled":
            X = self.X_scaled
        else:
            raise ValueError("Invalid stage")

        profile = {
            "shape": X.shape,
            "mean": float(X.values.mean()),
            "std": float(X.values.std()),
            "min": float(X.values.min()),
            "max": float(X.values.max()),
            "median": float(np.median(X.values)),
        }

        logger.info(f"Shape: {profile['shape']}")
        logger.info(f"Mean: {profile['mean']:.4f}")
        logger.info(f"Std: {profile['std']:.4f}")
        logger.info(f"Range: [{profile['min']:.4f}, {profile['max']:.4f}]")
        logger.info(f"Median: {profile['median']:.4f}")

        return profile

    def save_preprocessed_data(self):
        """Save preprocessed data for efficient reloading"""
        logger.info("Saving preprocessed data...")

        X_path = os.path.join(self.preprocessed_dir,
                             f"{self.dataset_name}_X_scaled.npy")
        y_path = os.path.join(self.preprocessed_dir,
                             f"{self.dataset_name}_y.npy")
        cols_path = os.path.join(self.preprocessed_dir,
                                f"{self.dataset_name}_genes.npy")

        np.save(X_path, self.X_scaled.values)
        np.save(y_path, self.y)
        np.save(cols_path, self.X_scaled.columns.values)

        logger.info(f"Saved X_scaled: {X_path}")
        logger.info(f"Saved y: {y_path}")
        logger.info(f"Saved gene names: {cols_path}")

    def load_preprocessed_data(self):
        """Load previously preprocessed data"""
        logger.info("Loading preprocessed data...")

        X_path = os.path.join(self.preprocessed_dir,
                             f"{self.dataset_name}_X_scaled.npy")
        y_path = os.path.join(self.preprocessed_dir,
                             f"{self.dataset_name}_y.npy")
        cols_path = os.path.join(self.preprocessed_dir,
                                f"{self.dataset_name}_genes.npy")

        X_scaled_array = np.load(X_path)
        y = np.load(y_path)
        gene_names = np.load(cols_path, allow_pickle=True)

        self.X_scaled = pd.DataFrame(X_scaled_array, columns=gene_names)
        self.y = y

        logger.info(f"Loaded X_scaled: {self.X_scaled.shape}")
        logger.info(f"Loaded y: {self.y.shape}")

        return self.X_scaled, self.y


# Example usage
if __name__ == "__main__":
    processor = GenomicDataProcessor(dataset_name="GSE19804")

    # Full preprocessing
    X, y = processor.preprocess_complete()

    # Data profiling at each stage
    processor.profile_data("raw")
    processor.profile_data("log")
    processor.profile_data("scaled")

    # Save for efficient reloading
    processor.save_preprocessed_data()

    print(f"\nPreprocessing complete!")
    print(f"X_scaled shape: {X.shape}")
    print(f"y shape: {y.shape}")
