import pandas as pd
import numpy as np

from src.constants import * 

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