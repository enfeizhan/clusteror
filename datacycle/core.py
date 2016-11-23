import pandas as pd


class Clusteror(object):
    def __init__(self, raw_data):
        self._raw_data = raw_data

    @classmethod
    def from_csv(cls, filepath, **kwargs):
        raw_data = pd.read_csv(filepath, **kwargs)
        return cls(raw_data)

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
