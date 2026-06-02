"""
Unified Preprocessing Pipeline
Replicates Jupyter notebook logic in modular, reusable format
Handles data ingestion, transformation, and artifact storage
"""

import logging
from pathlib import Path
import urllib.request
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

# Setup clean logger
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


class GenomicDataProcessor:
    """Centralized preprocessing pipeline for high-dimensional gene expression data"""

    def __init__(self, dataset_name="GSE19804", project_dir=""):
        """
        Initialize processor with dynamic metadata configurations.
        """
        self.dataset_name = dataset_name
        
        # Modern path handling using Pathlib
        self.project_dir = Path(project_dir) if project_dir else Path.cwd()
        self.datasets_dir = self.project_dir / "datasets"
        self.preprocessed_dir = self.project_dir / "preprocessed_datasets"

        self.datasets_dir.mkdir(parents=True, exist_ok=True)
        self.preprocessed_dir.mkdir(parents=True, exist_ok=True)

        # CENTRALIZED CONFIGURATION: The single source of truth for the entire project
        self.dataset_config = {
            "GSE42568": {
                "url": "https://ftp.ncbi.nlm.nih.gov/geo/series/GSE42nnn/GSE42568/matrix/GSE42568_series_matrix.txt.gz",
                "filename": "GSE42568_series_matrix.txt.gz",
                "cancer_type": "Breast Cancer",
                "n_cancer": 104,
                "n_normal": 17,
            },
            "GSE19804": {
                "url": "https://ftp.ncbi.nlm.nih.gov/geo/series/GSE19nnn/GSE19804/matrix/GSE19804_series_matrix.txt.gz",
                "filename": "GSE19804_series_matrix.txt.gz",
                "cancer_type": "Lung Cancer",
                "n_cancer": 60,
                "n_normal": 60,
            },
        }

        # Validate existence of requested target dataset configuration
        if self.dataset_name not in self.dataset_config:
            raise KeyError(
                f"Dataset '{self.dataset_name}' is not registered in dataset_config."
            )

        # Pipeline state placeholders
        self.X_raw = None
        self.X_log = None
        self.X_scaled = None
        self.y = None

    def download_dataset(self):
        """Automatically download dataset directly from GEO if not present locally"""
        config = self.dataset_config[self.dataset_name]
        filepath = self.datasets_dir / config["filename"]

        if filepath.exists():
            logger.info(f"Dataset target cache hit: {filepath}")
            return filepath

        logger.info(f"Downloading remote series matrix {self.dataset_name} from GEO FTP...")
        try:
            # Added a short timeout safeguard so download requests don't hang indefinitely
            urllib.request.urlretrieve(config["url"], filepath)
            logger.info(f"Downloaded successfully to: {filepath}")
            return filepath
        except Exception as e:
            logger.error(f"Network download transmission step failed: {e}")
            raise

    def load_data(self):
        """Load GEO series matrix, strip comments, and structure matrices"""
        logger.info(f"Loading experiment environment: {self.dataset_name}")

        filepath = self.download_dataset()
        config = self.dataset_config[self.dataset_name]

        # Load matrix, skipping metadata lines starting with '!'
        df = pd.read_csv(
            filepath, compression="infer", sep="\t", comment="!", index_col=0
        )

        # Transpose: features must map to columns, instances to rows
        self.X_raw = df.T

        # Create target labels: 1 = Cancer, 0 = Normal
        # Note: This assumes the data source maps all cancer rows before normal rows natively
        self.y = np.array([1] * config["n_cancer"] + [0] * config["n_normal"])

        logger.info(f"Data parsed: {self.X_raw.shape[0]} samples across {self.X_raw.shape[1]} features.")
        logger.info(f"Target vector mapped: {config['n_cancer']} Cancer | {config['n_normal']} Normal")

        return self.X_raw, self.y

    def apply_log_transformation(self):
        """Apply Log2 transformation to stabilize variance variance tracking"""
        if self.X_raw is None:
            raise ValueError("State trace uninitialized. Execute load_data() first.")

        logger.info("Applying monotonic Log2(x + 1) transformation step...")
        self.X_log = np.log2(self.X_raw + 1)
        
        # Optimization: Free raw data memory allocations if no longer required
        self.X_raw = None 
        return self.X_log

    def apply_standardization(self):
        """Apply Z-score standardization across expression profiles"""
        if self.X_log is None:
            raise ValueError("State trace uninitialized. Execute apply_log_transformation() first.")

        logger.info("Applying standardized Z-score scaling standard scale modifications...")
        scaler = StandardScaler()
        X_scaled_array = scaler.fit_transform(self.X_log)
        
        # Build DataFrame while preserving full structural components (index and columns)
        self.X_scaled = pd.DataFrame(
            X_scaled_array, index=self.X_log.index, columns=self.X_log.columns
        )

        # Optimization: Clear log matrix out of memory allocation footprints
        self.X_log = None 
        return self.X_scaled

    def preprocess_complete(self):
        """Execute linear cascade pipeline execution block"""
        logger.info("=" * 60)
        logger.info(f"INITIALIZING PIPELINE: {self.dataset_name}")
        logger.info("=" * 60)

        self.load_data()
        self.apply_log_transformation()
        self.apply_standardization()

        logger.info("=" * 60)
        logger.info("PIPELINE SEQUENCE EVALUATED SUCCESSFULLY")
        logger.info("=" * 60)

        return self.X_scaled, self.y

    def profile_data(self, stage="scaled"):
        """Generate mathematical data profiling snapshot from data frames"""
        logger.info(f"\n--- EXPERIMENTAL DATA PROFILE SUMMARY ({stage.upper()}) ---")

        if stage == "scaled":
            X = self.X_scaled
        elif stage == "raw" and self.X_raw is not None:
            X = self.X_raw
        elif stage == "log" and self.X_log is not None:
            X = self.X_log
        else:
            raise ValueError(f"Stage data matrix trace for '{stage}' is unavailable or was cleared from memory.")

        vals = X.values
        profile = {
            "shape": X.shape,
            "mean": float(vals.mean()),
            "std": float(vals.std()),
            "min": float(vals.min()),
            "max": float(vals.max()),
            "median": float(np.median(vals)),
        }

        logger.info(f"Dimensional Matrix Array Bounds: {profile['shape']}")
        logger.info(f"Global Expression Distribution Mean: {profile['mean']:.4f}")
        logger.info(f"Standard Error Deviation Matrix Delta: {profile['std']:.4f}")
        logger.info(f"Expression Profile Absolute Boundary Bounds: [{profile['min']:.4f}, {profile['max']:.4f}]")
        logger.info(f"Expression Profile Array Median Score: {profile['median']:.4f}\n")

        return profile

    def save_preprocessed_data(self):
        """Save preprocessed data matrices safely alongside structural index tags"""
        if self.X_scaled is None:
            raise ValueError("No scaled matrices stored. Can not serialize state outputs.")

        logger.info("Serializing analytical state targets to disk caches...")

        X_path = self.preprocessed_dir / f"{self.dataset_name}_X_scaled.npy"
        y_path = self.preprocessed_dir / f"{self.dataset_name}_y.npy"
        cols_path = self.preprocessed_dir / f"{self.dataset_name}_genes.npy"
        index_path = self.preprocessed_dir / f"{self.dataset_name}_samples.npy" # FIXED: Cache sample IDs!

        np.save(X_path, self.X_scaled.values)
        np.save(y_path, self.y)
        np.save(cols_path, self.X_scaled.columns.values)
        np.save(index_path, self.X_scaled.index.values)

        logger.info(f"Cache complete: Stored matrices and index arrays in {self.preprocessed_dir}")

    def load_preprocessed_data(self):
        """Load stored performance binaries to recover structured expression DataFrames"""
        logger.info("Checking disk signatures for preprocessed array binaries...")

        X_path = self.preprocessed_dir / f"{self.dataset_name}_X_scaled.npy"
        y_path = self.preprocessed_dir / f"{self.dataset_name}_y.npy"
        cols_path = self.preprocessed_dir / f"{self.dataset_name}_genes.npy"
        index_path = self.preprocessed_dir / f"{self.dataset_name}_samples.npy"

        # Explicitly checking paths via Pathlib metrics
        if not (X_path.exists() and y_path.exists() and cols_path.exists() and index_path.exists()):
            raise FileNotFoundError("Cached file signatures missing on storage volumes.")

        X_scaled_array = np.load(X_path)
        self.y = np.load(y_path)
        gene_names = np.load(cols_path, allow_pickle=True)
        sample_ids = np.load(index_path, allow_pickle=True)

        # Reconstruct DataFrame with structural indexes intact
        self.X_scaled = pd.DataFrame(
            X_scaled_array, index=sample_ids, columns=gene_names
        )

        logger.info(f"Data state restored successfully: {self.X_scaled.shape}")
        return self.X_scaled, self.y


if __name__ == "__main__":
    # Test operational trace validation block
    processor = GenomicDataProcessor(dataset_name="GSE19804")
    X, y = processor.preprocess_complete()
    processor.profile_data("scaled")
    processor.save_preprocessed_data()