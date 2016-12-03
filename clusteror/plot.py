'''
Plotting tools relevant for illustrating and comparing clustering results
can be found in this module.
'''
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
        filepath=None,
        **kwargs
        ):
    '''
    Plot the distribution of a two dimensional data against clustering groups
    in a scatter plot.

    A point represents an instance in the dataset. Points in a same cluster
    are painted with a same colour.

    This tool is useful to check the clustering impact in this two-dimensional
    sub-space.

    Parameters
    ----------

    two_dim_data: Pandas DataFrame
        A dataframe with two columns. The first column goes to the x-axis,
        and the second column goes to the y-axis.
    labels: list, Pandas Series, Numpy Array, or any iterable
        The segment label for each sample in ``two_dim_data``.
    markers: list
        Marker names for each group.
    bbox_to_anchor: tuple
        Instruction to placing the legend box relative to the axes. Details
        refer to ``Matplotlib`` document.
    colors: list, default None
        Colours for each group. Use equally distanced colours on colour map
        if not supplied.
    figsize: tuple
        Figure size (width, height).
    xlim: tuple
        X-axis limits.
    ylim: tuple
        Y-axis limits.
    alpha: float, between 0 and 1
        Marker transparency. From 0 to 1: from transparent to opaque.
    loc: int
        The corner of the legend box to anchor. Details refer to ``Matplotlib``
        document.
    grid: boolean, default True
        Show grid.
    show: boolean, default True
        Show figure in pop-up windows if true. Save to files if False.
    filepath: str
        File name to saving the plot. Must be assigned a valid filepath if
        ``show`` is False.
    **kwargs: keyword arguments
        Other keyword arguemnts passed on to ``matplotlib.pyplot.scatter``.

    Note
    ----

    Instances in a same cluster does not necessarily assemble together in
    all two dimensional sub-spaces. There can be possibly no clustering
    capaility for certain features. Additionally certain features play a
    secondary role in clustering as having less importance in
    ``field_importance`` in ``clusteror`` module.
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
        assert filepath
        plt.savefig(filepath)


def hist_plot_one_dim_group_data(
        one_dim_data,
        labels,
        bins=11,
        colors=None,
        figsize=(10, 6),
        xlabel='Dimension Reduced Data',
        ylabel='Occurance',
        bbox_to_anchor=(1.01, 1),
        loc=2,
        grid=True,
        show=True,
        filepath=None,
        **kwargs):
    '''
    Plot the distribution of a one dimensional numerical data in a histogram.
    This tool is useful to check the clustering impact in this one-dimensional
    sub-space.

    Parameters
    ----------

    one_dim_data: list, Pandas Series, Numpy Array, or any iterable
        A sequence of data. Each element if for an instance.
    labels: list, Pandas Series, Numpy Array, or any iterable
        The segment label for each sample in ``one_dim_data``.
    bins: int or iterable
        If an integer, bins - 1 bins created or a list of the delimiters.
    colors: list, default None
        Colours for each group. Use equally distanced colours on colour map
        if not supplied.
    figsize: tuple
        Figure size (width, height).
    xlabel: str
        Plot xlabel.
    ylabel: str
        Plot ylabel.
    bbox_to_anchor: tuple
        Instruction to placing the legend box relative to the axes. Details
        refer to ``Matplotlib`` document.
    loc: int
        The corner of the legend box to anchor. Details refer to ``Matplotlib``
        document.
    grid: boolean, default True
        Show grid.
    show: boolean, default True
        Show figure in pop-up windows if true. Save to files if False.
    filepath: str
        File name to saving the plot. Must be assigned a valid filepath if
        ``show`` is False.
    **kwargs: keyword arguments
        Other keyword arguemnts passed on to ``matplotlib.pyplot.scatter``.

    Note
    ----

    Instances in a same cluster does not necessarily assemble together in
    all one dimensional sub-spaces. There can be possibly no clustering
    capaility for certain features. Additionally certain features play a
    secondary role in clustering as having less importance in
    ``field_importance`` in ``clusteror`` module.
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
    plt.xlabel(xlabel, size=17)
    plt.ylabel(ylabel, size=17)
    if grid:
        plt.grid()
    if show:
        plt.show()
    else:
        assert filepath
        plt.savefig(filepath)


def group_occurance_plot(
        one_dim_data,
        cat_label,
        labels,
        group_label,
        colors=None,
        figsize=(10, 6),
        bbox_to_anchor=(1.01, 1),
        loc=2,
        grid=True,
        show=True,
        filepath=None,
        **kwargs):
    '''
    Plot the distribution of a one dimensional **ordinal or categorical** data
    in a bar chart. This tool is useful to check the clustering impact in this
    one-dimensional sub-space.

    Parameters
    ----------

    one_dim_data: list, Pandas Series, Numpy Array, or any iterable
        A sequence of data. Each element if for an instance.
    cat_label: str
        Field name will be used for the one dimensional data.
    labels: list, Pandas Series, Numpy Array, or any iterable
        The segment label for each sample in one_dim_data.
    group_label: str
        Field name will be used for the cluster ID.
    colors: list, default None
        Colours for each category existing in this one dimensional data.
        Default colour scheme used if not supplied.
    figsize: tuple
        Figure size (width, height).
    bbox_to_anchor: tuple
        Instruction to placing the legend box relative to the axes. Details
        refer to ``Matplotlib`` document.
    loc: int
        The corner of the legend box to anchor. Details refer to ``Matplotlib``
        document.
    grid: boolean, default True
        Show grid.
    show: boolean, default True
        Show figure in pop-up windows if true. Save to files if False.
    filepath: str
        File name to saving the plot. Must be assigned a valid filepath if
        ``show`` is False.
    **kwargs: keyword arguments
        Other keyword arguemnts passed on to ``matplotlib.pyplot.scatter``.

    Note
    ----

    Instances in a same cluster does not necessarily assemble together in
    all one dimensional sub-spaces. There can be possibly no clustering
    capaility for certain features. Additionally certain features play a
    secondary role in clustering as having less importance in
    ``field_importance`` in ``clusteror`` module.
    '''
    if not isinstance(one_dim_data, pd.core.series.Series):
        one_dim_data = pd.Series(one_dim_data)
    df = pd.DataFrame({cat_label: one_dim_data, group_label: labels})
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
        assert filepath
        plt.savefig(filepath)
