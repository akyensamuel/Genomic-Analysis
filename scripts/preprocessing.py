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

import gzip
import logging
import re
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
    # Dataset registry - single source of truth for the whole project
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
            # GSE42568's normal samples are an independently sampled
            # comparison group, not matched to specific cancer patients
            # (Chapter 3, Section 2.2) -> no patient grouping applies.
            "paired_samples": False,
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
            # Confirmed from the series matrix !Sample_title line: sample
            # titles are "Lung Cancer <id>T" / "Lung Normal <id>N", where
            # <id> is shared between a patient's tumour and paired normal
            # sample (e.g. "Lung Cancer 2T" and "Lung Normal 2N" are the
            # same patient). Cross-validation must keep both members of a
            # pair in the same fold (see _extract_sample_groups below).
            "paired_samples": True,
        },
    }

    def __init__(self, dataset_name: str = "GSE42568", project_dir: str = ""):
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
        self.groups: np.ndarray | None = None  # patient/case ID per sample, or None

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

        logger.info(f"Downloading {self.dataset_name} from GEO FTP ...")
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
        logger.info(f"Loading {self.dataset_name} ...")
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

        # Patient/case grouping, for datasets with matched tumour-normal
        # pairs (e.g. GSE19804). Extracted from the raw !Sample_title line,
        # which pd.read_csv above discarded via comment="!", so this line
        # is re-read directly from the same file.
        if config.get("paired_samples", False):
            self.groups = self._extract_sample_groups(filepath)
            if self.groups is not None and len(self.groups) == self.X_raw.shape[0]:
                logger.info(
                    f"Extracted {len(set(self.groups))} unique patient/case "
                    f"groups for {self.dataset_name} (paired_samples=True)."
                )
            else:
                logger.warning(
                    f"paired_samples=True for {self.dataset_name} but group "
                    "extraction failed or length mismatch; falling back to "
                    "ungrouped cross-validation. Verify the !Sample_title "
                    "parsing regex against this series' actual title format."
                )
                self.groups = None
        else:
            self.groups = None

        logger.info(
            f"Loaded: {self.X_raw.shape[0]} samples x {self.X_raw.shape[1]} features | "
            f"{config['n_cancer']} cancer / {config['n_normal']} normal"
        )
        return self.X_raw, self.y

    @staticmethod
    def _extract_sample_groups(filepath: Path) -> np.ndarray | None:
        """
        Parse the !Sample_title line from a GEO series matrix file and
        derive a patient/case group ID for each sample, in column order.

        Verified against GSE19804's actual format: titles are
        "Lung Cancer <id>T" / "Lung Normal <id>N", e.g. "Lung Cancer 2T"
        and "Lung Normal 2N" share patient ID "2". The regex below pulls
        the first run of digits out of each quoted title as the group ID,
        which is robust to the T/N suffix and the "Cancer"/"Normal" label
        but assumes each title contains exactly one numeric ID shared by
        a sample's pair. Returns None (rather than raising) if the line
        is missing or no titles matched, so callers can fall back to
        ungrouped cross-validation instead of crashing.
        """
        title_line = None
        with gzip.open(filepath, "rt", encoding="utf-8", errors="replace") as f:
            for line in f:
                if line.startswith("!Sample_title"):
                    title_line = line
                    break

        if title_line is None:
            return None

        # Titles are tab-separated, double-quoted strings after the
        # "!Sample_title" tag itself.
        titles = [t.strip().strip('"') for t in title_line.strip().split("\t")[1:]]
        ids = []
        for title in titles:
            match = re.search(r"(\d+)", title)
            if match is None:
                return None
            ids.append(match.group(1))

        if not ids:
            return None

        return np.array(ids)

    # Raw values with a maximum below this threshold are treated as already
    # being on a log2 scale (RMA/GCRMA-style GEO series matrices typically
    # range ~0-16), and the log2(x+1) transform below is skipped rather than
    # applied a second time. Raw values above this threshold are treated as
    # linear-scale (MAS5-style) intensities, for which log2(x+1) is applied.
    # Empirically confirmed for this project's two datasets: GSE19804's raw
    # range is [3.04, 14.89] and GSE42568's is [2.31, 16.05] -- both already
    # log2-scale, so log2(x+1) is skipped for both. This constant, not a
    # per-dataset flag, is what decides it, so any dataset added later to
    # DATASET_CONFIG is handled correctly without needing a manual toggle.
    LOG_SCALE_MAX_THRESHOLD: float = 20.0

    def apply_log_transformation(self) -> pd.DataFrame:
        """
        Apply log2(x + 1) transformation to stabilise variance -- but only
        if the raw values are not already on a log scale. Applying this
        transform to data that is already log2-scale (as both GSE19804 and
        GSE42568 were empirically found to be, via the raw-stage profiling
        step in __main__) would compress an already-appropriately-scaled
        variable a second time, which is incorrect.
        """
        if self.X_raw is None:
            raise ValueError("Run load_data() first.")

        raw_max = float(self.X_raw.values.max())
        if raw_max < self.LOG_SCALE_MAX_THRESHOLD:
            logger.info(
                f"Raw max value ({raw_max:.4f}) is below the log-scale "
                f"threshold ({self.LOG_SCALE_MAX_THRESHOLD}); data appears "
                "already log2-scale (RMA/GCRMA-style). Skipping log2(x+1) "
                "-- using raw values as-is for the 'log' pipeline stage."
            )
            self.X_log = self.X_raw
        else:
            logger.info(
                f"Raw max value ({raw_max:.4f}) exceeds the log-scale "
                f"threshold ({self.LOG_SCALE_MAX_THRESHOLD}); data appears "
                "linear-scale (MAS5-style). Applying log2(x + 1) ..."
            )
            self.X_log = np.log2(self.X_raw + 1)

        self.X_raw = None  # free memory
        return self.X_log

    def apply_standardization(self) -> pd.DataFrame:
        """Z-score standardise across all features."""
        if self.X_log is None:
            raise ValueError("Run apply_log_transformation() first.")

        logger.info("Applying Z-score standardisation ...")
        scaler = StandardScaler()
        X_scaled_arr = scaler.fit_transform(self.X_log)

        self.X_scaled = pd.DataFrame(
            X_scaled_arr,
            index=self.X_log.index,
            columns=self.X_log.columns,
        )
        # NOTE: self.X_log is intentionally NOT freed here (unlike the
        # original implementation). svm_classifier.py needs the
        # pre-standardization log-stage matrix (via save_log_stage_data /
        # load_log_stage_data) so it can fit StandardScaler fold-locally
        # instead of using this full-dataset-fit version. Callers that only
        # need X_scaled and want to free memory can del processor.X_log
        # themselves once save_log_stage_data() has been called.
        return self.X_scaled

    def preprocess_complete(self) -> tuple[pd.DataFrame, np.ndarray]:
        """Run the full pipeline: load -> log-transform -> standardise."""
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
    # Persistence - .npy cache (used by svm_classifier)
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

        logger.info(f"Saving preprocessed cache for {self.dataset_name} ...")

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

        NOTE: this returns the FULL-DATASET standardized matrix (fit on all
        samples). It is appropriate for full-dataset, descriptive
        characterization (e.g. feature_selection.py standalone mode / the
        Chapter 4 method-comparison tables), which does not evaluate
        held-out predictive performance. It is NOT appropriate as the
        direct input to a cross-validated classifier, because the scaler
        was fit using every sample, including whichever ones later become
        the held-out test fold. For that use case, call
        load_log_stage_data() instead and fit StandardScaler fold-locally
        (see svm_classifier.py, which does exactly this).
        """
        logger.info(f"Loading preprocessed cache for {self.dataset_name} ...")

        cache_dir = self.preprocessed_dir / self.dataset_name
        X_path     = cache_dir / f"{self.dataset_name}_X_scaled.npy"
        y_path     = cache_dir / f"{self.dataset_name}_y.npy"
        cols_path  = cache_dir / f"{self.dataset_name}_genes.npy"
        index_path = cache_dir / f"{self.dataset_name}_samples.npy"

        missing = [p for p in (X_path, y_path, cols_path, index_path) if not p.exists()]
        if missing:
            raise FileNotFoundError(
                f"Cache incomplete - missing file(s): {[str(m) for m in missing]}"
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
    # Persistence - LOG-STAGE cache (pre-scaling; used by svm_classifier.py
    # for cross-validated evaluation, so that standardization can be fit
    # fold-locally instead of on the full dataset)
    # ------------------------------------------------------------------
    def save_log_stage_data(self) -> None:
        """
        Serialise the log-transformed (NOT yet standardized) matrix to
        preprocessed_datasets/<dataset_name>/<dataset_name>_X_log.npy,
        alongside the y/genes/samples arrays already written by
        save_preprocessed_data(). This is the correct input for any
        pipeline that must fit StandardScaler fold-locally. Also persists
        self.groups (patient/case IDs), if extracted, for use with
        StratifiedGroupKFold on paired datasets such as GSE19804.
        """
        if self.X_log is None:
            raise ValueError(
                "No log-stage data to save. Call apply_log_transformation() "
                "and save before apply_standardization() frees self.X_log."
            )

        logger.info(f"Saving log-stage cache for {self.dataset_name} ...")

        out_dir = self.preprocessed_dir / self.dataset_name
        out_dir.mkdir(parents=True, exist_ok=True)

        np.save(out_dir / f"{self.dataset_name}_X_log.npy", self.X_log.values)
        np.save(out_dir / f"{self.dataset_name}_y.npy", self.y)
        np.save(out_dir / f"{self.dataset_name}_genes.npy", self.X_log.columns.values)
        np.save(out_dir / f"{self.dataset_name}_samples.npy", self.X_log.index.values)
        if self.groups is not None:
            np.save(out_dir / f"{self.dataset_name}_groups.npy", self.groups)
            logger.info(f"Saved patient/case groups ({len(set(self.groups))} unique).")

        logger.info(f"Log-stage cache saved to: {out_dir}")

    def load_log_stage_data(self) -> tuple[pd.DataFrame, np.ndarray, np.ndarray | None]:
        """
        Load the log-stage (pre-standardization) cache written by
        save_log_stage_data(). Raises FileNotFoundError if the core X/y/
        genes/samples files are missing. Returns (X_log, y, groups), where
        groups is None if this dataset has no patient/case grouping (e.g.
        GSE42568) or the groups file was never written.
        """
        logger.info(f"Loading log-stage cache for {self.dataset_name} ...")

        cache_dir = self.preprocessed_dir / self.dataset_name
        X_path     = cache_dir / f"{self.dataset_name}_X_log.npy"
        y_path     = cache_dir / f"{self.dataset_name}_y.npy"
        cols_path  = cache_dir / f"{self.dataset_name}_genes.npy"
        index_path = cache_dir / f"{self.dataset_name}_samples.npy"
        groups_path = cache_dir / f"{self.dataset_name}_groups.npy"

        missing = [p for p in (X_path, y_path, cols_path, index_path) if not p.exists()]
        if missing:
            raise FileNotFoundError(
                f"Log-stage cache incomplete - missing file(s): {[str(m) for m in missing]}"
            )

        y = np.load(y_path)
        X_log = pd.DataFrame(
            np.load(X_path),
            index=np.load(index_path, allow_pickle=True),
            columns=np.load(cols_path, allow_pickle=True),
        )
        groups = np.load(groups_path, allow_pickle=True) if groups_path.exists() else None

        logger.info(
            f"Log-stage cache loaded: {X_log.shape}"
            + (f" with {len(set(groups))} patient/case groups" if groups is not None else " (no groups)")
        )
        return X_log, y, groups

    # ------------------------------------------------------------------
    # Persistence - CSV export (used by feature_selection standalone mode)
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

    # Run stages manually (rather than via preprocess_complete()) so each
    # intermediate stage can be profiled before it is freed from memory.
    # The "raw" profile in particular answers the thesis Chapter 3 question
    # of whether the downloaded GEO matrix is already on a log scale: if
    # these raw values are already confined to a small range (e.g. ~0-16),
    # log2(x+1) is not appropriate and should be skipped.
    processor.load_data()
    processor.profile_data("raw")

    processor.apply_log_transformation()
    processor.profile_data("log")

    processor.apply_standardization()
    processor.profile_data("scaled")

    # Full-dataset-standardized cache: used for descriptive, non-CV
    # characterization (e.g. feature_selection.py standalone mode).
    processor.save_preprocessed_data()
    processor.export_csv()
    # Log-stage (pre-standardization) cache: used by svm_classifier.py so
    # that StandardScaler can be fit fold-locally during cross-validation.
    processor.save_log_stage_data()