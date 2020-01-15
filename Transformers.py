import numpy as np
import pandas as pd
from sklearn.base import TransformerMixin
from sklearn.preprocessing import OrdinalEncoder, OneHotEncoder, StandardScaler


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

    def fit(self, x, y=None, **fitparams):
        assert (isinstance(x, np.ndarray)), "input must have numpy ndarray data type"
        return self

    def transform(self, x):
        return pd.DataFrame(x, columns=self.columns)


class DeleteColumn(TransformerMixin):
    """
    DeleteColumn deletes specified column,s from pandas DataFrame
    """

    def __init__(self, column_names_to_delete):
        """

        param: column_names_to_delete: list of column names to delete specified as strings
        """

        assert (isinstance(column_names_to_delete, list)), "argument: column_names_to_delete must be of type list"
        self.column_names_to_delete = column_names_to_delete
        check_string_type(column_names_to_delete)

    def fit(self, x, y=None, **fitparams):
        assert (isinstance(x, pd.DataFrame)), "input must be of type pandas.DataFrame"
        check_string_type(x.columns.tolist())
        return self

    def transform(self, x):
        for column in self.column_names_to_delete:
            x.pop(column)
        return x


class DeleteNullRows(TransformerMixin):
    """
    Delete rows from the data set where null occur in the column
    """

    def __init__(self, column_names):
        """
        param: column_names: list of columns where we check for null values
        """

        assert (isinstance(column_names, list)), "argument: column_names must be of type list"
        check_string_type(column_names)
        self.column_names = column_names

    def fit(self, x, y=None, **fitparams):
        assert (isinstance(x, pd.DataFrame)), "input must be of type pandas.DataFrame"
        check_string_type(x.columns.tolist())
        return self

    def transform(self, x):
        x.dropna(subset=self.column_names, inplace=True)
        return x


class OrdinalEncoderWrapper(TransformerMixin):
    """
    Transform categorical data into numerical data
    """

    def __init__(self, column_names):
        """
        param:column_names: list of columns that will be transformed
        """
        assert (isinstance(column_names, list)), "argument: column_names must be of type list"
        check_string_type(column_names)
        self.column_names = column_names

    def fit(self, x, y=None, **kwargs):
        self.enc = OrdinalEncoder().fit(x[self.column_names])
        return self

    def transform(self, x):
        x[self.column_names] = self.enc.transform(x[self.column_names])
        return x


class OneHotEncoderWrapper(TransformerMixin):
    """
    Transform categorical data into one hot encoders
    """

    def __init__(self, column_names):
        """
        param: column_names: list of columns that will be transformed
        """

        assert (isinstance(column_names, list)), "argument: column_names must be of type list"
        check_string_type(column_names)
        self.column_names = column_names
        self.enc_list = []

    def fit(self, x, y=None, **kwargs):
        for column in self.column_names:
            self.enc_list.append(OneHotEncoder().fit(x[[column]]))
        return self

    def transform(self, x):
        # Transform data

        for encoder, column_name in zip(self.enc_list, self.column_names):
            data = encoder.transform(x[[column_name]]).toarray()

            new_column_names = encoder.get_feature_names()
            for i in range(len(new_column_names)):
                x[new_column_names[i]] = data[:, i]

        for column_name in self.column_names:
            x.pop(column_name)
        return x

    @staticmethod
    def __remove_columns(x, list_of_columns):
        for column in list_of_columns:
            x[column].pop()


class StandardScalerWrapper(TransformerMixin):

    def __init__(self, column_names):
        """
        param:column_names: list of columns that will be transformed
        """

        assert (isinstance(column_names, list)), "argument: column_names must be of type list"
        check_string_type(column_names)
        self.column_names = column_names
        self.scalers = []

    def fit(self, x, y=None, **kwargs):
        for column in self.column_names:
            self.scalers.append(StandardScaler().fit(x[[column]]))
        return self

    def transform(self, x, y=None, **kwargs):
        for scaler, column_name in zip(self.scalers, self.column_names):
            data = scaler.transform(x[[column_name]])
            x[column_name] = data
        return x


def check_string_type(list_of_values):
    for value in list_of_values:
        assert (isinstance(value, str))
