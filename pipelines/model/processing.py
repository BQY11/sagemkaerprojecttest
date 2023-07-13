import argparse
import os
import requests
import tempfile

import pandas as pd

from sklearn import preprocessing
import numpy as np


if __name__ == "__main__":
    base_dir = "/opt/ml/processing"

    df = pd.read_csv(
        f"{base_dir}/input/trainval.csv",
        header=None
    )
    df = df.sample(frac=1, random_state=13).reset_index(drop=True)
    train_size =  int(df.shape[0] * 0.8)
    train, validation = df.iloc[: train_size], df.iloc[train_size:]

    os.makedirs(f"{base_dir}/train", mode=0o777, exist_ok=True)
    os.makedirs(f"{base_dir}/validation", mode=0o777, exist_ok=True)

    
    train.to_csv(f"{base_dir}/train/train.csv", header=False, index=False)
    validation.to_csv(
        f"{base_dir}/validation/validation.csv", header=False, index=False
    )