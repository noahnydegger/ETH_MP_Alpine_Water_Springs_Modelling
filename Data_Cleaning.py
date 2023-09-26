import pandas as pd  # data processing
import numpy as np  # data processing
import os  # interaction with operating system


def keep_only_valid_data(df):
    return df[df.valid]
