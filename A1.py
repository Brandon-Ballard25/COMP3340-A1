import pandas as pd
import numpy as np

from src.constants import *
from src.task_1 import *
from src.task_2 import *
from src.task_3 import *
from src.task_4 import *
from src.task_5 import *

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