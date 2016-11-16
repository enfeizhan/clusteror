import numpy as np


class DataStore(object):
    def __init__(self):
        self.raw_data = np.nan
        self.cleaned_data = np.nan
        self.id_col_name = None
        self.transactions = np.nan

    @property
    def raw_data(self):
        return self._raw_data

    @raw_data.setter
    def raw_data(self, raw_data):
        self._raw_data = raw_data

    @property
    def cleaned_data(self):
        return self._cleaned_data

    @cleaned_data.setter
    def cleaned_data(self, cleaned_data):
        self._cleaned_data = cleaned_data

    @property
    def id_col_name(self):
        return self._id_col_name

    @id_col_name.setter
    def id_col_name(self, id_col_name):
        self._id_col_name = id_col_name

    @property
    def transactions(self):
        return self._transactions

    @transactions.setter
    def transactions(self, transactions):
        self._transactions = transactions
