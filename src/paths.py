"""Repository paths shared by notebooks and helpers."""

from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
RAW = DATA / "raw"
PROCESSED = DATA / "processed"
OUTPUTS = ROOT / "outputs"
FIGURES = OUTPUTS / "figures"
TABLES = OUTPUTS / "tables"
MODELS = OUTPUTS / "models"
SPLITS = ROOT / "splits"

SPLIT_PATH = SPLITS / "split.npz"
ROLES_PATH = OUTPUTS / "column_roles.json"
GATE_PATH = OUTPUTS / "graph_gate.json"
CONTRACT_PATH = OUTPUTS / "question_contract.json"
FEATURE_LOG_PATH = OUTPUTS / "feature_log.json"
ANALYSIS_PATH = PROCESSED / "analysis_frame.parquet"
FEATURES_PATH = PROCESSED / "features.parquet"
MODEL_MATRIX_PATH = PROCESSED / "model_matrix.parquet"
ALIAS_PATH = OUTPUTS / "alias_map.csv"
ANSWERS_PATH = OUTPUTS / "answers.md"
COMPARISON_PATH = OUTPUTS / "model_comparison.csv"
BUNDLE_PATH = MODELS / "selected_model.joblib"


def ensure_dirs() -> None:
    """Create the folders notebooks write into."""
    for path in (RAW, PROCESSED, OUTPUTS, FIGURES, TABLES, MODELS, SPLITS):
        path.mkdir(parents=True, exist_ok=True)
