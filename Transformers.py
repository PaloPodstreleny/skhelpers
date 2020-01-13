from sklearn.base import TransformerMixin
import pandas as pd
import numpy as np


class DataFrameWrapper(TransformerMixin):
    """
    DataFrameWrapper transform numpy ndarray into pandas DataFrame object
    """

    def __init__(self, list_column_names=None):
        """

        :param list_column_names: (default=None) list containing column names for DataFrame.
               Lenght of column names and columns in ndarray must be same.
               If  list_column_names == None: No column names will be used for DataFrame creation
        """
        if list_column_names is not None:
            assert (isinstance(list_column_names, list)), "list_column_names must have data type None or list "

        self.columns = list_column_names

    def fit(self, x, y=None):
        assert (isinstance(x, np.ndarray)), "input must have numpy ndarray data type"
        return self

    def transform(self, x):
        return pd.DataFrame(x, columns=self.columns)
