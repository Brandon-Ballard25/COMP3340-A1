import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent

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

STRING_COLUMNS = [
  "quality"
]

NORMAL_RANGE = {
  "density": (0.9, 1.1)
}


# Task 1: Data Overview

def task_1(df):
  print("\nTask 1: Data Overview")

  print("\nFirst 5 rows:")
  print(df.head())

  print("\nDataset information:")
  df.info()

  print("\nDescriptive statistics:")
  print(df.describe(include="all"))

  print("\nColumn names:")
  print(df.columns.tolist())

  print("\nData types:")
  print(df.dtypes)

  print("\nDataset shape:")
  print(df.shape)

  print("\nMissing values:")
  print(df.isna().sum())

  print("\nDuplicate rows:")
  print(df.duplicated().sum())


# Task 2: Data Cleaning

def remove_empty_rows(df):
  return df.dropna(how="all").copy()


def strip_and_lowercase_column_names(df):
  df.columns = df.columns.str.strip().str.lower()
  return df

def strip_and_lowercase_values(df, columns):
  for column in columns:
    df[column] = (
      df[column]
      .astype("string")
      .str.strip()
      .str.lower()
    )

  return df
  

def clean_quality_labels(df):
  corrections = {"meedium": "medium", "hhigh": "high"}

  df["quality"] = df["quality"].replace(corrections)
  df["quality"] = df["quality"].astype("object").replace(WINE_QUALITY_TO_NUM_MAPPING)
  return df


def convert_to_numeric(df, columns):
  for column in columns:
    df[column] = pd.to_numeric(
      df[column],
      errors="coerce"
    )

  return df


def replace_negatives_with_nan(df, columns):
  for column in columns:
    df.loc[df[column] < 0, column] = np.nan

  return df


def replace_out_of_bounds_with_nan(df, ranges):
  for column, (min_val, max_val) in ranges.items():
    df.loc[(df[column] < min_val) | (df[column] > max_val), column] = np.nan

  return df


def fill_missing_with_median(df, columns):
  for column in columns:
    median = df[column].median()
    df[column] = df[column].fillna(median)

  return df


def task_2(df):
  print("\nTask 2: Data Cleaning")

  # Print empty rows:
  empty_rows = df.isna().all(axis=1).sum()
  print("\nCompletely empty rows:", empty_rows)

  # Clean pipeline. 
  df = remove_empty_rows(df)
  df = strip_and_lowercase_column_names(df)
  df = strip_and_lowercase_values(df, STRING_COLUMNS)
  df = clean_quality_labels(df)
  df = convert_to_numeric(df, NUMERIC_COLUMNS)
  df = replace_negatives_with_nan(df, NON_NEGATIVE_COLUMNS)
  df = replace_out_of_bounds_with_nan(df, NORMAL_RANGE)

  print("\nNaN values after validation:")
  print(df.isna().sum())

  # Fill missing values with median for numeric columns
  df = fill_missing_with_median(df, NUMERIC_COLUMNS)

  # Duplicate rows check and removal
  duplicate_count = df.duplicated().sum()
  print("\nDuplicate rows found:", duplicate_count)
  df = (df.drop_duplicates().reset_index(drop=True))

  print("\n========== CLEANED DATA ==========")

  print("\nFirst 5 rows:")
  print(df.head())

  print("\nDescriptive statistics:")
  print(df.describe(include="all"))

  df.to_csv(CLEAN_FILE, index=False)
  print(f"\nCleaned dataset saved to: {CLEAN_FILE}")

'''
  stores a scatter plot with ax built with the
  provided params
  x: column for x a-xis
  y: column for y y-xis
  colors: color map
  df: dataset
  ax: container for figure
'''
def create_scatter_plot(x,y,colors,df,ax):
  ax.scatter(df[x],df[y], c=colors)
  ax.set_title(f"{x.replace('_', " ")} vs {y.replace('_', " ")}")
  ax.set_xlabel(x.replace('_', " "))
  ax.set_ylabel(y.replace('_', " "))
  ax.set_ylim(bottom=None, top=df[y].max()*1.01)

def task_3(df):
  print("\nTask 3: Exploratory Visualisation")

  clean_df = pd.read_csv(CLEAN_FILE)  
  color_map = {
      0:'red',
      1:'orange',
      2:'green',
    }
  #create colour map
  colors = [color_map[col] for col in clean_df['quality']]
  _, ax = plt.subplots(2, 2)

  #plot shows the relationship visualizes density and acidity
  create_scatter_plot('density','fixed_acidity', {'blue'}, clean_df, ax[0,0])
  #plot adds color depth to plot
  create_scatter_plot('density','fixed_acidity', colors, clean_df, ax[0,1])
  #plot shows a compressed data set
  create_scatter_plot('alcohol','free_sulfur_dioxide', colors, clean_df, ax[1,0])

  #rebuilds colormap for reduced data frame
  colors = [color_map[col] for col in clean_df['quality'][clean_df['free_sulfur_dioxide'] < 100]]
  #build scatter plot with no extreme values from sulfur
  create_scatter_plot('alcohol','free_sulfur_dioxide', colors, clean_df[clean_df['free_sulfur_dioxide'] < 100], ax[1,1])
  
  #fix height spacing
  plt.subplots_adjust(hspace=0.5) 
  
  plt.show(block=False)


def task_4(df):
  print("\nTask 4: Feature Magnitudes and Scaling")
  
  # Apply z-score normalization to numeric columns
  df[NUMERIC_COLUMNS] = (
    df[NUMERIC_COLUMNS] - df[NUMERIC_COLUMNS].mean()
) / df[NUMERIC_COLUMNS].std()
  
  # Verify mean and std values for all columns
  for col in NUMERIC_COLUMNS:
    print("Column: {}  Mean = {}  Std = {}".format(col,df[col].mean().round(2), df[col].std().round(2)))
    if (df[col].mean().round(2),df[col].std().round(2)) != (0,1):
      print("Error standardizing values")
  print("Z-score normalization applied to numeric data")
  

def task_5(df):
  print("\nTask 5: Feature Engineering")

  # PCA is applied to the cleaned numeric features, not to the quality label.
  X_wine = df[NUMERIC_COLUMNS] 
  y_wine = df["quality"]

  # Standardise deez features b4 fitting PCA.
  wine_scaler = StandardScaler()
  wine_scaler.fit(X_wine)
  X_wine_scaled = wine_scaler.transform(X_wine)

  # Fit PCA with all components so that the variance contribution of each
  # will let variance contrib be inspected for all components
  wine_pca = PCA(n_components=None)
  wine_pca.fit(X_wine_scaled)
  wine_scores = wine_pca.transform(X_wine_scaled)

  print("Explained-variance ratio:", wine_pca.explained_variance_ratio_.round(4))
  print("Shape of the scores:", wine_scores.shape)

  # get all individual and cumulative contribs aswell
  wine_individual = wine_pca.explained_variance_ratio_
  wine_cumulative = np.cumsum(wine_individual)
  component_numbers = np.arange(1, len(wine_individual) + 1)

  wine_contribution = pd.DataFrame({
    "component": [f"PC{i}" for i in component_numbers],
    "individual_contribution": wine_individual,
    "cumulative_contribution": wine_cumulative
  })

  print("\nIndividual explained-variance ratios:")
  print(wine_contribution[["component", "individual_contribution"]].round(4))
  print("\nCumulative explained-variance ratios:")
  print(wine_contribution[["component", "cumulative_contribution"]].round(4))

  # Determine the number of components needed to retain at least 98% of the variance.
  components_to_keep = np.argmax(wine_cumulative >= 0.98) + 1
  print(f"\nComponents needed to retain at least 98%: {components_to_keep}")
  print(
    "Cumulative contribution retained:",
    round(wine_cumulative[components_to_keep - 1], 4)
  )

  # graphing stuff I grabbed from the tutorial
  plt.figure(figsize=(6, 4))
  plt.bar(component_numbers, wine_individual * 100, label="Individual")
  plt.plot(
    component_numbers,
    wine_cumulative * 100,
    marker="o",
    color="darkorange",
    label="Cumulative"
  )
  plt.xticks(component_numbers)
  plt.ylim(0, 105)
  plt.xlabel("Principal component")
  plt.ylabel("Explained variance (%)")
  plt.title("PCA contribution for the cleaned wine data")
  plt.legend()
  plt.legend()
  plt.savefig(BASE_DIR / "data" / "output" / "wine_pca_contribution.png",
              dpi=300,
              bbox_inches="tight")
  plt.close() 


  plt.show()

  # Keep the smallest number of components that reaches the 98% threshold.
  wine_scores_reduced = wine_scores[:, :components_to_keep]

  df_pca = pd.DataFrame(
    wine_scores_reduced,
    columns=[f"PC{i}" for i in range(1, components_to_keep + 1)]
  )
  df_pca["quality"] = y_wine.to_numpy()
  df_pca.to_csv(PCA_FILE, index=False)

  print(f"\nPCA-transformed data saved: {PCA_FILE}")
  return df_pca


# additional thoughts"
# - add all columns to NON_NEGATIVE_COLUMNS
# - convert low, high, medium into numeric values.

if __name__ == "__main__":
  df = pd.read_csv(RAW_FILE)
  CLEAN_FILE.parent.mkdir(parents=True, exist_ok=True)

  print("Starting tasks...")

  # Data audit and initial findings 
  task_1(df)

  # Cleaning methods and justification 
  task_2(df)

  # Exploratory visualisation
  task_3(df)

  # Feature magnitudes and scaling
  task_4(df)

  # Feature Engineering
  task_5(pd.read_csv(CLEAN_FILE))

  print("All tasks completed.")
