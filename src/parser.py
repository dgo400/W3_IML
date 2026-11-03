import numpy as np
import pandas as pd
from scipy.io import arff


def load_arff(file_path):
    """Turn .arff into pandas DataFrame object with features (X) and labels (y)."""
    data, meta = arff.loadarff(file_path)
    df = pd.DataFrame(data)

    # Nominal attributes are read as bytes, so we convert them to text
    for column in df.columns:
        if df[column].dtype == object:
            df[column] = df[column].str.decode("utf-8")

    # Missing nominal values appear as "?", numeric ones are already NaN
    df = df.replace("?", np.nan)

    # The class is the last attribute, we keep it apart from the data
    X = df.iloc[:, :-1]
    y = df.iloc[:, -1]
    return X, y


if __name__ == "__main__":
    for name in ["iris", "satimage", "horse-colic"]:
        X, y = load_arff("data/" + name + ".arff")
        print(name, X.shape, "Missing values:", X.isna().sum().sum())
