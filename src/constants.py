from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent

RAW_FILE = BASE_DIR / "data" / "input" / "wine_data.csv"
CLEAN_FILE = BASE_DIR / "data" / "output" / "wine_cleaned.csv"
PCA_FILE = BASE_DIR / "data" / "output" / "wine_pca_transformed.csv"

WINE_QUALITY_TO_NUM_MAPPING = {
  "low": 0,
  "medium": 1,
  "high": 2
}

NON_NEGATIVE_COLUMNS = [
  "fixed_acidity",
  "residual_sugar",
  "free_sulfur_dioxide",
  "density",
  "alcohol",
]

NUMERIC_COLUMNS = [
  "fixed_acidity",
  "residual_sugar",
  "free_sulfur_dioxide",
  "density",
  "alcohol",
]

NUMERIC_COLUMNS = [
  "fixed_acidity",
  "residual_sugar",
  "free_sulfur_dioxide",
  "density",
  "alcohol",
]

NORMAL_RANGE = {
  "density": (0.9, 1.1)
}
