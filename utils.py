import numpy as np
import base64
import cv2
import pandas as pd


def decode_array(s: str) -> np.array:
    b = base64.b64decode(s.encode('utf8'))
    return np.frombuffer(b, dtype=np.float64)


def load_rbg(path: str) -> np.ndarray:
    return cv2.cvtColor(cv2.imread(path), cv2.COLOR_BGR2RGB)


def get_from_csv(csv_path: str) -> pd.DataFrame:
    dataframe = pd.read_csv(csv_path, index_col=0)
    dataframe['vector'] = dataframe['vector'].map(decode_array)
    return dataframe
