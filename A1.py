import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

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
  df["quality"] = df["quality"].replace(WINE_QUALITY_TO_NUM_MAPPING)
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


def task_3(df):
  clean_df = pd.read_csv(CLEAN_FILE)
  print("\nTask 3: Exploratory Visualisation")
  #compare_df_columns_scatter(clean_df)
  #compare_attributes_per_quality(clean_df)
  #three_dimensional_graphs('alcohol','residual_sugar','quality', clean_df)
  #bar_graph_quality(clean_df)

'''
Produces figures that show the values of each column
for different qualities of wine
'''
def compare_attributes_per_quality(df):
  #loop columns
  for column in df.columns:
      if column == "quality":
        continue
      #produce graph
      bar_graph_quality_single_attribute(df, column)

"""
Groups data by the quality of wine, then calculates
the mean values for each attribute within each wine
quality and displays it as a bar graph
"""
def bar_graph_quality_single_attribute(df, column):
    quality_labels = {
      0: "Low",
      1: "Medium",
      2: "High"
    }
    quality_mean = df.groupby(['quality']).mean().rename(index=quality_labels)
    
    quality_mean[column].plot(kind='bar')
    
    plt.xlabel('Wine quality')
    plt.ylabel(f'{column}')
    plt.title('Mean chemistry of different wine qualities')
    plt.show()


"""
Groups data by the quality of wine, then calculates
the mean values for each attribute within each wine
quality and displays it as a bar graph
"""
def bar_graph_quality(df):
    quality_labels = {
      0: "Low",
      1: "Medium",
      2: "High"
    }
    quality_mean = df.groupby(['quality']).mean().rename(index=quality_labels)
    
    quality_mean.plot(kind='bar')
    
    plt.xlabel('Wine attributes')
    plt.ylabel('Attribute values')
    plt.title('Mean chemistry of different wine qualities')
    plt.show()

'''
x y z: column names for data that will be visualized
df: dataframe
produces a 3d scatter plot using provided data
'''
def three_dimensional_graphs(x,y,z, df):
  fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
  ax.scatter(df[x], df[y],df[z])
  ax.set_xlabel(x)
  ax.set_ylabel(y)
  ax.set_zlabel(z)
  plt.tight_layout()
  plt.show()

"""
compares columns to every other column within
the dataset. produces a 2 column figure with multiple
graphs for a comparison between features
"""
def compare_df_columns_scatter(df):
  #determine the length of each column
  df_len = len(df.columns)
  if(df_len%2!= 0):
    #maintains whole numbers for indexing
    df_len += 1

  #iterate over columns
  for x_count, x in enumerate(df.columns):
      fig, ax = plt.subplots((df_len//2),2)
      fig_index = 0;
      second_col = True
      #iterate over columns to compare to previous column
      for y_count, y in enumerate(df.columns,0):
        
        if y == x:
          #dont produce graphs that compare the same column
          continue
        #decide which column of the figure to image
        if(second_col):
          #first column
          ax[fig_index,0].scatter(df[x],df[y])
          ax[fig_index,0].set_title(f"Wine chemistry {x.replace('_', " ")} vs {y.replace('_', " ")}")
          ax[fig_index,0].set_xlabel(x)
          ax[fig_index,0].set_ylabel(y)

        else:
          #second column
          ax[fig_index,1].scatter(df[x],df[y])
          ax[fig_index,1].set_title(f"Wine chemistry {x.replace('_', " ")} vs {y.replace('_', " ")}")
          ax[fig_index,1].set_xlabel(x)
          ax[fig_index,1].set_ylabel(y)

        fig_index += 1
        if fig_index == df_len/2:
          fig_index = 0
          second_col = not second_col
  plt.suptitle(f'{y} vs remaining features')
  
  plt.tight_layout()
  plt.show()

def task_4(df):
  print("\nTask 4: Feature Magnitudes and Scaling")

def task_5(df):
  print("\nTask 5: Feature Engineering")


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
  task_5(df)

  print("All tasks completed.")