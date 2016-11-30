import matplotlib.pyplot as plt
import pandas as pd
import numpy as np


def scatter_plot_two_dim_group_data(
        two_dim_data,
        labels,
        markers=None,
        colors=None,
        figsize=(10, 6),
        xlim=None,
        ylim=None,
        alpha=0.8,
        bbox_to_anchor=(1.01, 1),
        loc=2,
        grid=True,
        show=True,
        filename=None,
        **kwargs
        ):
    '''
    Plot the distribution of a two dimensional data in a scatter plot.

    two_dim_data: pandas dataframe
    A dataframe with two columns. The first column goes to the x-axis, and the
    second column goes to the y-axis.

    labels: list, pandas series, or numpy array
    The segment label for each sample in two_dim_data.

    markers:
    Marker names for each group.

    bbox_to_anchor: tuple

    '''
    assert isinstance(two_dim_data, pd.core.frame.DataFrame)
    assert two_dim_data.shape[1] == 2, 'Two_dim_data must have two columns!'
    if isinstance(labels, pd.core.series.Series):
        labels = labels.values
    grouped = two_dim_data.groupby(labels)
    n_groups = grouped.ngroups
    # there should be enough markers
    if markers is not None:
        error_msg = 'There should be one marker for each group!'
        assert len(markers) == n_groups, error_msg
    # get color for each group from the spectrum
    if colors is None:
        colors = plt.cm.Spectral(np.linspace(0, 1, n_groups))
    plt.figure(figsize=figsize)
    ax = plt.subplot(111)
    if markers is None:
        # do a for loop to plot one by one
        # if markers not given, default circles
        for (name, group), color in zip(grouped, colors):
            ax.scatter(
                x=group.values[:, 0],
                y=group.values[:, 1],
                color=color,
                label=str(name),
                alpha=alpha,
                **kwargs)
    else:
        for (name, group), color, marker in zip(grouped, colors, markers):
            ax.scatter(
                x=group.values[:, 0],
                y=group.values[:, 1],
                color=color,
                marker=marker,
                label=str(name),
                alpha=alpha,
                ax=ax,
                **kwargs)
    # place the legend at the right hand side of the chart
    plt.legend(bbox_to_anchor=bbox_to_anchor, loc=loc)
    # get the axes names
    x_label, y_label = tuple(two_dim_data.columns)
    plt.xlabel(x_label, size=17)
    plt.ylabel(y_label, size=17)
    # get lim for x and y axes
    if xlim is None:
        xlim = (two_dim_data.iloc[:, 0].min(), two_dim_data.iloc[:, 0].max())
    if ylim is None:
        ylim = (two_dim_data.iloc[:, 1].min(), two_dim_data.iloc[:, 1].max())
    plt.xlim(xlim)
    plt.ylim(ylim)
    if grid:
        plt.grid()
    if show:
        plt.show()
    else:
        assert filename
        plt.savefig(filename)


def hist_plot_one_dim_group_data(
        one_dim_data,
        labels,
        bins=11,
        colors=None,
        figsize=(10, 6),
        bbox_to_anchor=(1.01, 1),
        loc=2,
        grid=True,
        show=True,
        filename=None,
        **kwargs):
    '''
    Plot the distribution of the one dimensional reduced data in a histogram.
    The range of data is always from zero to one.

    one_dim_data: list, pandas series, or numpy array
    The one dimensional reduced data in a one dimensional data type.

    labels: list, pandas series, or numpy array
    The segment label for each sample in one_dim_data.

    bins: integer or array
    If an integer, bins - 1 bins between minimum and maximum of the subclass,
    or an list of the delimiters.
    '''
    if not isinstance(one_dim_data, pd.core.series.Series):
        one_dim_data = pd.Series(one_dim_data)
    if isinstance(labels, pd.core.series.Series):
        labels = labels.values
    grouped = one_dim_data.groupby(labels)
    n_groups = grouped.ngroups
    # get color for each group from the spectrum
    if colors is None:
        colors = plt.cm.Spectral(np.linspace(0, 1, n_groups))
    plt.figure(figsize=figsize)
    ax = plt.subplot(111)
    # if bins is a integer create bins between 0 and 1
    if isinstance(bins, int):
        bins = np.linspace(-1, 1, bins)
    # do a for loop to plot one by one
    for (name, group), color in zip(grouped, colors):
        ax.hist(
            group.values,
            bins=bins,
            color=color,
            label=str(name),
            alpha=0.5,
            **kwargs
        )
    # place the legend at the right hand side of the chart
    plt.legend(bbox_to_anchor=bbox_to_anchor, loc=loc)
    plt.xlabel('Dimension Reduced Data', size=17)
    plt.ylabel('Occurence', size=17)
    if grid:
        plt.grid()
    if show:
        plt.show()
    else:
        assert filename
        plt.savefig(filename)


def group_occurance_plot(
        data,
        cat_label,
        labels,
        group_label,
        colors=None,
        figsize=(10, 6),
        bbox_to_anchor=(1.01, 1),
        loc=2,
        grid=True,
        show=True,
        filename=None,
        **kwargs):
    '''
    Plot the distribution of the one dimensional reduced data in a histogram.
    The range of data is always from zero to one.

    one_dim_data: list, pandas series, or numpy array
    The one dimensional reduced data in a one dimensional data type.

    labels: list, pandas series, or numpy array
    The segment label for each sample in one_dim_data.

    bins: integer or array
    If an integer, bins - 1 bins between minimum and maximum of the subclass,
    or an list of the delimiters.
    '''
    if not isinstance(data, pd.core.series.Series):
        data = pd.Series(data)
    df = pd.DataFrame({cat_label: data, group_label: labels})
    df_to_plot = df.pivot_table(
        index=group_label,
        columns=cat_label,
        aggfunc=len
    )
    plt.figure(figsize=figsize)
    ax = plt.subplot(111)
    df_to_plot.plot.bar(color=colors, ax=ax, **kwargs)
    plt.legend(bbox_to_anchor=bbox_to_anchor, loc=loc)
    if grid:
        plt.grid()
    if show:
        plt.show()
    else:
        assert filename
        plt.savefig(filename)
