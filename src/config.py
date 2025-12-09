import os
from pathlib import Path

DATASET_ROOT_PATH = str(Path(__file__).parent.parent / "data")
os.makedirs(DATASET_ROOT_PATH, exist_ok=True)

DATASET_TRAIN = str(Path(DATASET_ROOT_PATH) / "PAKDD2010_Modeling_Data.txt")

DATASET_TEST = str(Path(DATASET_ROOT_PATH) / "PAKDD2010_Prediction_Data.txt")

DATASET_DESCRIPTION = str(
    Path(DATASET_ROOT_PATH) / "PAKDD2010_VariablesList.XLS"
)