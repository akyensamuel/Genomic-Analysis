"""
Unified Preprocessing Pipeline
================================
Replicates Jupyter notebook logic in modular, reusable format.
Handles data ingestion, transformation, and artifact storage.

Standalone usage
----------------
    python preprocessing.py [dataset_name]

    dataset_name  One of the keys in dataset_config (default: GSE19804).
                  Runs the full pipeline, profiles the result, and saves
                  the preprocessed cache to disk.

Fix log (vs original)
---------------------
- Removed module-level basicConfig call; caller is responsible for
  configuring logging. basicConfig is called only inside __main__ so
  importing this module never reconfigures the root logger.
- Added export_csv() to produce a labelled CSV that feature_selection.py
  (standalone mode) and any other script can consume directly, closing
  the data-exchange gap between the .npy cache and the CSV-based loader.
- project_dir now defaults to the directory containing this file rather
  than Path.cwd(), so the correct project root is used regardless of
  the working directory at import / instantiation time.
"""

from __future__ import annotations

import logging
import sys
import urllib.request
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

logger = logging.getLogger(__name__)


class GenomicDataProcessor:
    """Centralised preprocessing pipeline for high-dimensional gene expression data."""

    # ------------------------------------------------------------------
    # Dataset registry — single source of truth for the whole project
    # ------------------------------------------------------------------
    DATASET_CONFIG: dict = {
        "GSE42568": {
            "url": (
                "https://ftp.ncbi.nlm.nih.gov/geo/series/GSE42nnn/"
                "GSE42568/matrix/GSE42568_series_matrix.txt.gz"
            ),
            "filename": "GSE42568_series_matrix.txt.gz",
            "cancer_type": "Breast Cancer",
            "n_cancer": 104,
            "n_normal": 17,
        },
        "GSE19804": {
            "url": (
                "https://ftp.ncbi.nlm.nih.gov/geo/series/GSE19nnn/"
                "GSE19804/matrix/GSE19804_series_matrix.txt.gz"
            ),
            "filename": "GSE19804_series_matrix.txt.gz",
            "cancer_type": "Lung Cancer",
            "n_cancer": 60,
            "n_normal": 60,
        },
    }

    def __init__(self, dataset_name: str = "GSE19804", project_dir: str = ""):
        """
        Parameters
        ----------
        dataset_name : str
            Key into DATASET_CONFIG.
        project_dir : str
            Absolute path to the project root.  Defaults to the directory
            containing this file (not cwd) so the correct paths are used
            regardless of where Python is invoked from.
        """
        if dataset_name not in self.DATASET_CONFIG:
            raise KeyError(
                f"Dataset '{dataset_name}' is not registered in DATASET_CONFIG. "
                f"Available: {list(self.DATASET_CONFIG)}"
            )

        self.dataset_name = dataset_name

        # Default to the file's own directory rather than cwd so paths are
        # stable regardless of where the script is run from.
        self.project_dir = (
            Path(project_dir) if project_dir else Path(__file__).resolve().parent.parent
        )
        self.datasets_dir = self.project_dir / "datasets"
        self.preprocessed_dir = self.project_dir / "preprocessed_datasets"

        self.datasets_dir.mkdir(parents=True, exist_ok=True)
        self.preprocessed_dir.mkdir(parents=True, exist_ok=True)

        # Expose a per-instance view of the config for callers that iterate it
        self.dataset_config = self.DATASET_CONFIG

        # Pipeline state
        self.X_raw: pd.DataFrame | None = None
        self.X_log: pd.DataFrame | None = None
        self.X_scaled: pd.DataFrame | None = None
        self.y: np.ndarray | None = None

    # ------------------------------------------------------------------
    # Download
    # ------------------------------------------------------------------
    def download_dataset(self) -> Path:
        """Download the GEO series matrix if not already cached locally."""
        config = self.DATASET_CONFIG[self.dataset_name]
        filepath = self.datasets_dir / config["filename"]

        if filepath.exists():
            logger.info(f"Dataset already cached: {filepath}")
            return filepath

        logger.info(f"Downloading {self.dataset_name} from GEO FTP …")
        try:
            urllib.request.urlretrieve(config["url"], filepath)
            logger.info(f"Download complete: {filepath}")
        except Exception as exc:
            logger.error(f"Download failed: {exc}")
            raise

        return filepath

    # ------------------------------------------------------------------
    # Load & transform
    # ------------------------------------------------------------------
    def load_data(self) -> tuple[pd.DataFrame, np.ndarray]:
        """Load the GEO series matrix, transpose, and create the label vector."""
        logger.info(f"Loading {self.dataset_name} …")
        filepath = self.download_dataset()
        config = self.DATASET_CONFIG[self.dataset_name]

        # Skip metadata comment lines starting with '!'
        df = pd.read_csv(
            filepath, compression="infer", sep="\t", comment="!", index_col=0
        )

        # Transpose so rows = samples, columns = genes
        self.X_raw = df.T

        # Labels: 1 = Cancer, 0 = Normal
        # Assumes GEO ordering: all cancer samples come before normal samples.
        self.y = np.array(
            [1] * config["n_cancer"] + [0] * config["n_normal"], dtype=int
        )

        logger.info(
            f"Loaded: {self.X_raw.shape[0]} samples × {self.X_raw.shape[1]} features | "
            f"{config['n_cancer']} cancer / {config['n_normal']} normal"
        )
        return self.X_raw, self.y

    def apply_log_transformation(self) -> pd.DataFrame:
        """Apply log2(x + 1) transformation to stabilise variance."""
        if self.X_raw is None:
            raise ValueError("Run load_data() first.")

        logger.info("Applying log2(x + 1) transformation …")
        self.X_log = np.log2(self.X_raw + 1)
        self.X_raw = None  # free memory
        return self.X_log

    def apply_standardization(self) -> pd.DataFrame:
        """Z-score standardise across all features."""
        if self.X_log is None:
            raise ValueError("Run apply_log_transformation() first.")

        logger.info("Applying Z-score standardisation …")
        scaler = StandardScaler()
        X_scaled_arr = scaler.fit_transform(self.X_log)

        self.X_scaled = pd.DataFrame(
            X_scaled_arr,
            index=self.X_log.index,
            columns=self.X_log.columns,
        )
        self.X_log = None  # free memory
        return self.X_scaled

    def preprocess_complete(self) -> tuple[pd.DataFrame, np.ndarray]:
        """Run the full pipeline: load → log-transform → standardise."""
        logger.info("=" * 60)
        logger.info(f"PREPROCESSING PIPELINE: {self.dataset_name}")
        logger.info("=" * 60)

        self.load_data()
        self.apply_log_transformation()
        self.apply_standardization()

        logger.info("=" * 60)
        logger.info("PIPELINE COMPLETE")
        logger.info("=" * 60)

        return self.X_scaled, self.y

    # ------------------------------------------------------------------
    # Profiling
    # ------------------------------------------------------------------
    def profile_data(self, stage: str = "scaled") -> dict:
        """Log and return summary statistics for a pipeline stage."""
        stage_map = {
            "scaled": self.X_scaled,
            "raw": self.X_raw,
            "log": self.X_log,
        }
        X = stage_map.get(stage)
        if X is None:
            raise ValueError(
                f"Stage '{stage}' is unavailable (either not yet computed or "
                f"already freed from memory). Available stages: "
                f"{[k for k, v in stage_map.items() if v is not None]}"
            )

        vals = X.values
        profile = {
            "shape": X.shape,
            "mean": float(vals.mean()),
            "std": float(vals.std()),
            "min": float(vals.min()),
            "max": float(vals.max()),
            "median": float(np.median(vals)),
        }

        logger.info(f"--- DATA PROFILE ({stage.upper()}) ---")
        logger.info(f"  Shape  : {profile['shape']}")
        logger.info(f"  Mean   : {profile['mean']:.4f}")
        logger.info(f"  Std    : {profile['std']:.4f}")
        logger.info(f"  Range  : [{profile['min']:.4f}, {profile['max']:.4f}]")
        logger.info(f"  Median : {profile['median']:.4f}")

        return profile

    # ------------------------------------------------------------------
    # Persistence — .npy cache (used by svm_classifier)
    # ------------------------------------------------------------------
    def save_preprocessed_data(self) -> None:
        """
        Serialise the scaled matrix to four .npy files under
        preprocessed_datasets/<dataset_name>/.

        Files written
        -------------
        <name>_X_scaled.npy   – float array (n_samples, n_features)
        <name>_y.npy          – int array   (n_samples,)
        <name>_genes.npy      – object array of gene/feature names
        <name>_samples.npy    – object array of sample IDs
        """
        if self.X_scaled is None:
            raise ValueError("No scaled data to save. Run preprocess_complete() first.")

        logger.info(f"Saving preprocessed cache for {self.dataset_name} …")

        out_dir = self.preprocessed_dir / self.dataset_name
        out_dir.mkdir(parents=True, exist_ok=True)

        np.save(out_dir / f"{self.dataset_name}_X_scaled.npy", self.X_scaled.values)
        np.save(out_dir / f"{self.dataset_name}_y.npy", self.y)
        np.save(out_dir / f"{self.dataset_name}_genes.npy", self.X_scaled.columns.values)
        np.save(out_dir / f"{self.dataset_name}_samples.npy", self.X_scaled.index.values)

        logger.info(f"Cache saved to: {out_dir}")

    def load_preprocessed_data(self) -> tuple[pd.DataFrame, np.ndarray]:
        """
        Load the .npy cache written by save_preprocessed_data().

        Raises FileNotFoundError if any of the four expected files are missing.
        """
        logger.info(f"Loading preprocessed cache for {self.dataset_name} …")

        cache_dir = self.preprocessed_dir / self.dataset_name
        X_path     = cache_dir / f"{self.dataset_name}_X_scaled.npy"
        y_path     = cache_dir / f"{self.dataset_name}_y.npy"
        cols_path  = cache_dir / f"{self.dataset_name}_genes.npy"
        index_path = cache_dir / f"{self.dataset_name}_samples.npy"

        missing = [p for p in (X_path, y_path, cols_path, index_path) if not p.exists()]
        if missing:
            raise FileNotFoundError(
                f"Cache incomplete — missing file(s): {[str(m) for m in missing]}"
            )

        self.y = np.load(y_path)
        self.X_scaled = pd.DataFrame(
            np.load(X_path),
            index=np.load(index_path, allow_pickle=True),
            columns=np.load(cols_path, allow_pickle=True),
        )

        logger.info(f"Cache loaded: {self.X_scaled.shape}")
        return self.X_scaled, self.y

    # ------------------------------------------------------------------
    # Persistence — CSV export (used by feature_selection standalone mode)
    # ------------------------------------------------------------------
    def export_csv(self, out_path: Path | None = None) -> Path:
        """
        Write the scaled matrix plus a 'label' column to a CSV file so that
        feature_selection.py (standalone mode) can load it directly.

        The file is written to
            preprocessed_datasets/<dataset_name>/<dataset_name>.csv
        unless out_path is given explicitly.

        Returns the path of the written file.
        """
        if self.X_scaled is None:
            raise ValueError("No scaled data to export. Run preprocess_complete() first.")

        if out_path is None:
            out_dir = self.preprocessed_dir / self.dataset_name
            out_dir.mkdir(parents=True, exist_ok=True)
            out_path = out_dir / f"{self.dataset_name}.csv"

        df_out = self.X_scaled.copy()
        df_out.insert(0, "label", self.y)
        df_out.to_csv(out_path)

        logger.info(f"CSV exported to: {out_path}  ({df_out.shape})")
        return out_path


# ---------------------------------------------------------------------------
# Standalone entry point
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s  %(levelname)-8s  %(message)s",
        datefmt="%H:%M:%S",
    )

    dataset_name = sys.argv[1] if len(sys.argv) > 1 else "GSE19804"

    processor = GenomicDataProcessor(dataset_name=dataset_name)
    X, y = processor.preprocess_complete()
    processor.profile_data("scaled")
    processor.save_preprocessed_data()
    processor.export_csv()
