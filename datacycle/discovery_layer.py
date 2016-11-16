import sys
from data_layer import DataStore
import pandas as pd
import numpy as np
from utils import not_null
from fpgrowth import FPTree
from fpgrowth import generate_association_rules


class DataDiscovery(DataStore):
    def check_health(self, df, filename):
        '''
        Check health of the dataset.
        '''
        diagnosis = pd.DataFrame({})
        for col in df.columns:
            series = df[col]
            # show how many null's and how many not_null's
            is_known = not_null(series)
            try:
                # check if there are known values
                known_num = is_known[True]
            except KeyError:
                # sometims all are nulls
                known_num = 0
            try:
                # check if there are unknown values
                unknown_num = is_known[False]
            except KeyError:
                # hopefully there are no unknowns
                unknown_num = 0
            # get percentage for knowns and unknowns
            known_prop = 100 * known_num / (known_num + unknown_num)
            unknown_prop = 100 - known_prop
            # what are the values in the column?
            value_counts = series.value_counts()
            # list the first two most frequent ones
            try:
                v1 = value_counts.index[0]
                t1 = value_counts.iloc[0]
            except IndexError:
                v1 = None
                t1 = np.nan
            try:
                v2 = value_counts.index[1]
                t2 = value_counts.iloc[1]
            except IndexError:
                v2 = None
                t2 = np.nan
            # get the data range
            max_value = series.dropna().max()
            min_value = series.dropna().min()
            # check what data type it is
            data_type = series.dtype
            if np.issubdtype(data_type, np.number):
                # for numbers get more stats
                first_quar = series.quantile(0.25)
                median = series.quantile(0.5)
                third_quar = series.quantile(0.75)
                avg = series.mean()
                stdev = series.std()
                if np.issubdtype(data_type, np.integer):
                    data_type = 'integer'
                elif np.issubdtype(data_type, np.inexact):
                    data_type = 'float'
            else:
                data_type = 'text'
                first_quar = np.nan
                median = np.nan
                third_quar = np.nan
                avg = np.nan
                stdev = np.nan
            row = pd.DataFrame({
                'column name': [col],
                'data type': [data_type],
                'non-null entry number': [known_num],
                'non-null entry percentage (%)': [known_prop],
                'null entry number': [unknown_num],
                'null entry percentage (%)': [unknown_prop],
                'unique values': [len(value_counts)],
                'most frequent entry': [v1],
                'most frequent entry showing times': [t1],
                'second most frequent entry': [v2],
                'second most frequent entry showing times': [t2],
                'minimum': [min_value],
                'first_quartile': [first_quar],
                'median': [median],
                'third_quartile': [third_quar],
                'maximum': [max_value],
                'average': [avg],
                'standard deviation': [stdev]},
                columns=[
                    'column name',
                    'data type',
                    'non-null entry number',
                    'non-null entry percentage (%)',
                    'null entry number',
                    'null entry percentage (%)',
                    'unique values',
                    'most frequent entry',
                    'most frequent entry showing times',
                    'second most frequent entry',
                    'second most frequent entry showing times',
                    'minimum',
                    'first_quartile',
                    'median',
                    'third_quartile',
                    'maximum',
                    'average',
                    'standard deviation',
                    ])
            diagnosis = diagnosis.append(row)
        diagnosis.to_csv(filename, float_format='%.2f', index=False)

    def find_anormalies(self):
        pass

    def check_correlation(
            self,
            df,
            filename,
            maximum=0.6,
            ):
        '''
        Compute pairwise correlation of numerical columns.

        Parameters
        filename : object
            If to save a copy, the filename for the copy.
        maximum:
            A message printed if correlation is larger than `maximum`.
        Returns
        -------
        DataFrame for the pairwise correlation.

        Notes
        ------
        High correlation coefficients trigger a message printed. The
        correlation coefficients are saved in a csv file by default.
        '''
        # select only numerical columns
        num_cols = df.select_dtypes(include=['number'])
        # calculate pairwise correlation coefficients.
        corr = num_cols.corr()
        # wipe lower triangle values
        corr.values[np.tril_indices_from(corr)] = np.nan
        # trigger printing when large correlation
        for row_ind, row in corr.iterrows():
            for col_ind, value in row.iteritems():
                if value > maximum:
                    print('The correlation between {v1} '.format(v1=row_ind) +
                          'and {v2} is '.format(v2=col_ind) +
                          '{corr_value}, '.format(corr_value=float(value)) +
                          'larger than preset ' +
                          '{max_lim}.'.format(max_lim=float(maximum)))
        # save to file
        corr.to_csv(filename)
        return corr

    def find_transaction_patterns(
            self,
            transactions,
            threshold,
            filename=None):
        sys.setrecursionlimit(10000)
        tree = FPTree(transactions, threshold, None, None)
        patterns = tree.mine_patterns(threshold)
        # transform pattern dict into a dataframe
        # a string now is the pattern
        string_key_patterns = (
            self._tuplekeyeddict2dataframe(
                patterns,
                'Pattern',
                'Pattern Count'))
        if filename is not None:
            string_key_patterns.to_csv(filename, index=False)
        return string_key_patterns

    def find_association_rules(
            self,
            confidence_threshold,
            lower_support_threshold,
            threshold=1,
            transactions=None,
            patterns_filename=None,
            patterns=None,
            rules_filename=None):
        '''
        Find patterns and/or association rules.
        patterns_filename:
        File name to store patterns found given transactions.
        rules_filename:
        Fiel name to store association rules found given transaction or
        patterns.
        '''

        sys.setrecursionlimit(10000)
        if transactions is not None and patterns is None:
            tree = FPTree(transactions, threshold, None, None)
            patterns = tree.mine_patterns(threshold)
            if patterns_filename is not None:
                string_key_patterns = (
                    self._tuplekeyeddict2dataframe(
                        patterns,
                        'Pattern',
                        'Pattern Count'))
                string_key_patterns.to_csv(patterns_filename, index=False)
        rules = generate_association_rules(patterns, confidence_threshold,
                                           lower_support_threshold)
        to_dataframe = {
            'Antecedent': [],
            'Consequent': [],
            'Confidence': []}
        for antecedent, item in rules.items():
            # make sure antecedent are tuple of strings
            antecedent = [str(i) for i in antecedent]
            # make sure consequent are tuple of strings
            consequent = [str(i) for i in item[0]]
            # join to make strings
            antecedent = ','.join(antecedent)
            consequent = ','.join(consequent)
            to_dataframe['Antecedent'].append(antecedent)
            to_dataframe['Consequent'].append(consequent)
            to_dataframe['Confidence'].append(item[1])
        to_dataframe = pd.DataFrame(to_dataframe)
        to_dataframe = to_dataframe.reindex(
            columns=['Antecedent', 'Consequent', 'Confidence'])
        if patterns_filename is not None:
            to_dataframe.to_csv(rules_filename, index=False)
        return to_dataframe

    def _tuplekeyeddict2dataframe(self, tuplekeyeddict, key_name, value_name):
        string_key_dict = {}
        for key, item in tuplekeyeddict.items():
            # first make sure keys are tuple of strings
            key = [str(i) for i in key]
            # join the elements in key
            key = ','.join(key)
            string_key_dict[key] = item
        # transform dict to pandas series
        string_key_dict = pd.Series(string_key_dict)
        # transform to data frame and add headers
        string_key_dict = string_key_dict.reset_index()
        string_key_dict.columns = [key_name, value_name]
        return string_key_dict


if __name__ == '__main__':
    dd = DataDiscovery()
    transactions = [[1, 2, 5],
                    [2, 4],
                    [2, 3],
                    [1, 2, 4],
                    [1, 3],
                    [2, 3],
                    [1, 3],
                    [1, 2, 3, 5],
                    [1, 2, 3]]
    patterns = dd.find_transaction_patterns(transactions, 4)
    rules = dd.find_association_rules(
        transactions=transactions,
        threshold=4,
        confidence_threshold=0.51,
        lower_support_threshold=2)
