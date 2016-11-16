import os
import numpy as np
import pandas as pd
# import matplotlib.pyplot as plt
import pickle as pk
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from scipy.cluster.hierarchy import linkage
from scipy.cluster.hierarchy import dendrogram
from scipy.special import expit
from scipy.special import logit
from scipy.stats import chisquare
from scipy.stats import f_oneway
from preprocessing_layer import DataPreprocessing
from utils import confirm_proceed


class Segmentation(DataPreprocessing):
    def reduce_dimension(
            self,
            raw_data,
            cleaned_data,
            id_col_name,
            to_lower_dim_filename=None,
            field_weights=None,
            to_dim=1,
            batch_size=1000,
            training_epoch=10,
            verbose=True,
            patience=60,
            patience_increase=2,
            improvement_threshold=0.9995,
            ):
        '''
        Reduce dimension of datasets for further clustering in lower dimension
        space.
        dat: pandas dataframe
        Data in the form just before being taken sigmoid.

        id_col_name: string
        The id/pk column name.

        to_dim: int
        The lower dimension the datasets reduced to.

        clustering_features: list of strings
        The features to reduce.

        batch_size: int
        The size of minibatch in statistic gradient descent.

        training_epoch: int
        The number of epochs to train the neuronetwork.

        '''
        self.raw_data = raw_data
        self.id_col_name = id_col_name
        self.cleaned_data = cleaned_data
        if to_lower_dim_filename is not None:
            with open(to_lower_dim_filename, 'rb') as f:
                to_lower_dim = pk.load(f)
            self.one_dim_data = to_lower_dim(self.cleaned_data)
        else:
            self.da_reduce_dim(
                self.cleaned_data,
                field_weights=field_weights,
                to_dim=to_dim,
                batch_size=batch_size,
                training_epochs=training_epoch,
                verbose=verbose,
                patience=patience,
                patience_increase=patience_increase,
                improvement_threshold=improvement_threshold,
                )
            self.one_dim_data = self.to_lower_dim(self.cleaned_data)

    def add_segment(self, n_clusters=None, km_filename=None):
        '''
        Add segment column to existing raw dataset. Use K-Means model if
        provided, otherwise create a K-Means model.
        '''
        if km_filename is None:
            assert n_clusters is not None,\
                'n_clusters could not be None while km is None!'
            self.km = KMeans(n_clusters=n_clusters)
            self.km.fit(self.one_dim_data)
        else:
            with open(km_filename, 'rb') as f:
                self.km = pk.load(f)
        self.raw_data.loc[:, 'segment'] = self.km.predict(self.one_dim_data)
        self.raw_data = (
            self
            .raw_data
            .set_index(['segment', self.id_col_name]))

    def save_km(self, filename):
        with open(filename, 'wb') as f:
            pk.dump(self.km, f)
        print('K-Means saved in {}.'.format(filename))


# def hierarchical_clustering(
#         dat,
#         method='centroid',
#         metric='euclidean',
#         savefilename=None):
#     linkage_matrix = linkage(dat, method=method, metric=metric)
#     plt.figure()
#     ax = plt.subplot(111)
#     dendrogram(linkage_matrix)
#     plt.title('Hierarchical Clustering Dendrogram', size=18)
#     plt.xlabel('Samples', size=18)
#     plt.ylabel('Distance Between Clusters', size=18)
#     plt.setp(ax.get_xticklabels(), visible=False)
#     if savefilename is not None:
#         plt.savefig(savefilename)


def recover_centroids(
        dp,
        cluster_centers,
        ):
    recovered = dp.reconstruct(cluster_centers)
    recovered = logit(recovered)
    cluster_characters = pd.DataFrame(
        recovered,
        columns=dp.cleaned_data.columns)
    cluster_characters.index.name = 'segment'
    return cluster_characters


def find_clustering_drivers(
        dp,
        clusters,
        driver_detector_filename,
        driver_power_filename,
        decision_line=1e-2,
        ):
    '''
    Select strong features. Output a strong feature report embracing
    features characterising each clusters and overall good features.
    Output the driver power counts.

    Parameters
    -----------
    dp: dataframe
        A dataframe with numerical columns.
    clusters: series
        The clustering results from get_cluster method.
    p_value_decision_line: float, default: 1e-2
        Under which two clusters are considered distinct.
    driver_detector_filename: string
        Filename storing features differentiating two groups.
    driver_power_filename: string
        Filename storing driver power counts.

    Outputs
    -------
    A good feature report.
    Driver power counts.
    Returns
    -------
    Returns the strong features.
    '''
    df = (
        dp
        .raw_data
        .loc[:, dp.cleaned_data.columns])

    # get the clique in an iterator
    def get_cluster_id_pair(n_clusters):
        for l1 in range(n_clusters):
            for l2 in range(l1+1, n_clusters):
                yield l1, l2
    n_clusters = len(set(clusters))
    # create a dataframe to save drivers' power
    driver_powers = pd.DataFrame(
        0,
        index=df.columns,
        columns=range(n_clusters))
    # create a upper triangular matrix storing the drivers splitting
    # two clusters apart
    driver_detectors = pd.DataFrame(
        '',
        index=range(n_clusters),
        columns=range(n_clusters))
    # create a dataframe to save drivers' mean and median
    for col_name, col in df.iteritems():
        for cluster_id1, cluster_id2 in get_cluster_id_pair(n_clusters):
            # pick up the two clusters to compare
            c1 = col.loc[(clusters == cluster_id1).values]
            c2 = col.loc[(clusters == cluster_id2).values]
            p_value = f_oneway(c1.values, c2.values)[1]
            # if p_value small, add one to that col for the two clusters
            if p_value < decision_line:
                driver_powers.loc[col_name, [cluster_id1, cluster_id2]] += 1
                driver_detectors.loc[cluster_id1, cluster_id2] = (
                    driver_detectors.loc[cluster_id1, cluster_id2] +
                    ',' + col_name
                    if driver_detectors.loc[cluster_id1, cluster_id2] != ''
                    else col_name)
    driver_powers.to_csv(driver_power_filename)
    driver_detectors.to_csv(driver_detector_filename)


def analyse_categorical_variables(
        dp,
        cat_features,
        labels,
        chi_decision_line,
        flagged_to_output='flagged_categorical_variables.csv',
        ):
    '''
    Analyse the categorical variables. Chi-square test detects variables
    distinct from expected.

    Parameters
    ----------
    df: dataframe
        Dataframe of only categorical variables.
    n_rows: int
        Number of rows, i.e., number of clients.
    chi_decision_line: float, default: 5e-2
        The decision line below which a variable is considered as important.
    flagged_to_output: object, default: 'flagged_categorical_variables.csv'
        The filename for output csv file storing difference over
        the expectation ratio.

    Returns and outputs
    -------------------
    Output a csv file storing the (observed-expected)/expected ratio
    for variables that are flagged as important.
    '''
    df = dp.raw_data.loc[:, cat_features]
    n_rows = df.shape[0]
    n_clusters = len(set(labels.values))
    cluster_sizes = labels.value_counts()
    cluster_labels = cluster_sizes.index.tolist()
    # as in the loop below the .csv file will be added with new output
    # in each loop, it should be removed it already exists
    info = input(
        'Reports on categorical variables will be lost by proceeding!' +
        ' Would you like to proceed ([y]/n)?')
    confirm_proceed(info, 'No progress. Original file kept.')
    if os.path.exists(flagged_to_output):
        os.remove(flagged_to_output)
    for col_ind, (col_name, col) in enumerate(df.iteritems()):
        # setup expected;
        # the indices' and columns' order of the observed below should follow
        # the same order
        # universal distribution
        value_frac = col.value_counts()
        # before losing value_frac's indices, keep it for setting up observed
        values = value_frac.index
        # how many values kept
        n_values = len(values)
        # get fractions over the total number of rows
        value_frac = value_frac.values / n_rows
        # reshape it to n X 1 that can be matrix multiplied
        value_frac = value_frac.reshape((n_values, 1))
        expected = np.dot(
            value_frac,
            cluster_sizes.values.reshape((1, n_clusters)))
        # setup observed, to be sure observed and expected df have same
        # indices and columns
        observed = pd.DataFrame(0, index=values, columns=cluster_labels)
        for cluster_id in cluster_labels:
            observed.loc[:, cluster_id].update(
                col.loc[(labels == cluster_id).values].value_counts())
        # set up a df to store the ratio over expected if flagged
        index = pd.MultiIndex.from_product([[col_name], values])
        ratio_over_exp = pd.DataFrame(
            np.nan,
            index=index,
            columns=cluster_labels)
        # p-values calculated against each cluster
        chi_p_vals = chisquare(observed.values, expected)[1]
        for cluster_ind, p_val in enumerate(chi_p_vals):
            # ratio calculated, given p-value is below the decision line
            if p_val < chi_decision_line:
                ratio = (
                    (observed.iloc[:, cluster_ind] -
                     expected[:, cluster_ind]) /
                    (expected[:, cluster_ind] +
                     .00001)) * 100
            # ratio is (observed - expected)/expected
                ratio_over_exp.iloc[:, cluster_ind] = ratio.values
            # ignore occurrence smaller than 100 or 20% of total occurrence
            is_scarce = (
                (observed.iloc[:, cluster_ind] < 50) &
                (
                    observed.iloc[:, cluster_ind] <
                    cluster_sizes.iloc[cluster_ind]*.2)
                )
            ratio_over_exp.iloc[is_scarce.values, cluster_ind] = np.nan
        # add header only for the first time
        header = col_ind == 0
        ratio_over_exp.to_csv(
            flagged_to_output,
            mode='a',
            header=header,
            float_format='%.2f%%')


def cluster_report(
        dp,
        n_clusters,
        reportfile='report.csv'):
    cols_in_original_order = dp.raw_data.columns
    groupby_seg = dp.raw_data.groupby(level=0)
    minimum = groupby_seg.min()
    minimum.index = pd.MultiIndex.from_product(
        [range(n_clusters), 'minimum'],
        names=['segment', 'stats'])
    maximum = groupby_seg.max()
    maximum.index = pd.MultiIndex.from_product(
        [range(n_clusters), 'maximum'],
        names=['segment', 'stats'])
    deciles = []
    decile_names = [
        'first decile',
        'second decile',
        'third decile',
        'fourth decile',
        'fifth decile',
        'sixth decile',
        'seventh decile',
        'eighth decile',
        'ninth decile']
    for i, decile_name in enumerate(decile_names):
        decile = groupby_seg.quantile(0.1*(i+1))
        decile.index = pd.MultiIndex.from_product(
            [range(n_clusters), decile_name],
            names=['segment', 'stats'])
        deciles.append(decile)
    average = groupby_seg.mean()
    average.index = pd.MultiIndex.from_product(
        [range(n_clusters), 'average'],
        names=['segment', 'stats'])
    # make count the same form
    count = minimum.copy()
    count.loc[:, :] = 1
    count = count.mul(groupby_seg.size(), axis=0, level=0)
    count.index = pd.MultiIndex.from_product(
        [range(n_clusters), 'count'],
        names=['segment', 'stats'])
    stdev = groupby_seg.std()
    stdev.index = pd.MultiIndex.from_product(
        [range(n_clusters), 'stdev'],
        names=['segment', 'stats'])
    report = pd.concat([
        count,
        minimum,
        maximum,
        average,
        stdev,
        pd.concat(deciles)])
    report_index = pd.MultiIndex.from_product(
        [range(n_clusters),
         ['count', 'average', 'stdev', 'minimum'] +
         decile_names +
         ['maximum']],
        names=['segment', 'stats'])
    report = report.reindex(index=report_index, columns=cols_in_original_order)
    report.to_csv(reportfile)


if __name__ == '__main__':
    # ############# #
    # training flow #
    # ############# #
    # get data
    dat = pd.read_csv('test.tsv', sep='\t')
    print('Reading complete!')
    dat = dat.fillna(0)
    # reduce dimension to one and get the dimension-reducing model
    for_clustering = (
        dat
        .loc[:, ['bookings',
                 'unique_passengers',
                 'legs',
                 'passengers_per_booking',
                 'fare']].copy() * 1.0)
    # found value 0 for fare. free flight? add 1 dollar to avoid infinity
    for_clustering.loc[:, 'fare'] += 1.
    ss = StandardScaler()
    for_clustering = pd.DataFrame(
        np.tanh(ss.fit_transform(np.log(for_clustering))),
        columns=for_clustering.columns)
    # with open('ss.pickle', 'wb') as f:
    #     pk.dump(ss, f)
    seg = Segmentation()
    seg.reduce_dimension(
        dat,
        for_clustering,
        'emailaddress',
        to_lower_dim_filename=None,
        field_weights=[10, 1, 1, 4, 1],
        to_dim=1,
        batch_size=100,
        training_epoch=500,
        verbose=True,
        patience=60,
        patience_increase=2,
        improvement_threshold=0.995)
    # # pickle dimension-reducing model
    # seg.save_da_reduce_dim('reduce_to_one_dim.pickle')
    # # do clustering and get k-means
    # seg.add_segment(n_clusters=15)
    # # pickle k-means
    # seg.save_km('kmeans.pickle')
    # seg.raw_data.to_csv('segmented.csv')
#     # ############## #
#     # rescoring flow #
#     # ############## #
#     # get data
#     dat = pd.read_csv('clustering2013_2014.csv', nrows=100)
#     # customerised clean cleaning before next step. ignored here
#     # get dim-reduced data
#     seg = Segmentation()
#     seg.reduce_dimension(
#         dat,
#         dat.loc[:, ['bookings', 'passengers', 'unique_passengers', 'legs']],
#         'emailaddress',
#         to_lower_dim_filename='reduce_to_one_dim.pickle',
#         )
#     seg.add_segment(km_filename='kmeans.pickle')
#    # # hierarchical_clustering(pd.DataFrame(one_dim_dat), savefilename='h.png')
#    # analyse_categorical_variables(
#    #     dp,
#    #     ['bookings'],
#    #     pd.Series(km.labels_),
#    #     1e-2,
#    #     flagged_to_output='flagged_categorical_variables.csv',
#    #     )
#     # print('find features')
#     # find_clustering_drivers(
#     #     dp,
#     #     pd.Series(km.labels_),
#     #     'detector.csv',
#     #     'power.csv')
#     # recovered = recover_centroids(dp, km.cluster_centers_, ss)
#     # recovered.to_csv('recovered.csv')
#     # print('generating reports...')
#     # cluster_report(dp, 10, 'report1.csv')
