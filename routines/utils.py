# -*- coding: utf-8 -*-
import json
from pathlib import Path
from typing import List, Tuple, Dict

import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
from matplotlib.axes import Axes
from matplotlib.text import Text
from matplotlib.figure import Figure
from statsmodels.tsa.stattools import coint
from tqdm import tqdm

from routines.dataframes import dataframe_from_coint_dict, get_ratios_dataframe


def find_cointegrated_pairs(data: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray, List[Tuple]]:
    """
    This function will look for pairs of cointegrated stocks.

    :param data: pd.DataFrame
    :return:
    """
    n = data.shape[1]
    score_matrix = np.zeros((n, n))
    pvalue_matrix = np.ones((n, n))
    keys = data.keys()

    # We store the stock pairs that are likely to be cointegrated
    pairs = []

    progress_bar = tqdm(total=n**2/2, desc='Running cointegration tests...')

    for i in range(n):

        for j in range(i+1, n):

            S1: pd.Series = data[keys[i]]                           # values from first column
            S2: pd.Series = data[keys[j]]                           # values from second column
            result: Tuple = coint(S1, S2)                           # level of cointegration
            score: float = result[0]                                # t-score
            pvalue: float = result[1]                               # p-value
            score_matrix[i, j] = score                              # add coint scores to score_matrix
            pvalue_matrix[i, j] = pvalue                            # add pvalues to pvalue_matrix

            # if the p-value is less than the tresh, append to list
            if pvalue < 0.02:
                pairs.append((keys[i], keys[j]))

            progress_bar.update(1)

    progress_bar.close()
    return score_matrix, pvalue_matrix, pairs


def do_coint_tests(
        dataframe: pd.DataFrame,
        pvalue_threshold: float = 0.04,
        coint_dir: Path=Path('tmp')) -> Tuple[pd.DataFrame, pd.DataFrame, List]:
    """
    Perform coint test between all series.
    Also writes the nested dictionaries to file.

    By default, the function will write the nested dictionaries to:
        - scores.json
        - pvalue.json

    :param dataframe: pandas DataFrame, where each series is a stock
    :param pvalue_threshold: float value representing threshold for pvalue to be considered cointegrated
    :return: Tuple of pandas DataFrames --> (df_scores, df_pvalue)
    """

    # big dictionary to keep track of values
    scores_dict: Dict[str, Dict[str, float]] = {}
    pvalue_dict: Dict[str, Dict[str, float]] = {}
    pairs: List[Tuple[str, str]] = []

    # iter through columns
    num_columns = len(dataframe.columns)
    pbar = tqdm(total=int(num_columns**2/2)-int(num_columns/2), desc='Performing Cointegration tests between columns')

    for i, (name1, s1) in enumerate(dataframe.iteritems()):

        for j, (name2, s2) in enumerate(dataframe.iloc[:, i+1:].iteritems()):

            # get coint results between two columns
            score, pvalue, _ = coint(s1, s2)

            # add scores and pvalues to dictionaries
            update_coint_dict(scores_dict, score, name1, name2)
            update_coint_dict(pvalue_dict, pvalue, name1, name2)

            # if the p-value is less than the tresh, append to list
            if pvalue < pvalue_threshold:
                pairs.append((name1, name2))

            pbar.update(1)

    coint_dir.resolve()
    coint_dir.mkdir(parents=True, exist_ok=True)

    with (coint_dir / 'scores.json').open('w') as f:
        json.dump(scores_dict, f, indent=4)

    # with open('scores.json', 'w') as f:
    #     json.dump(scores_dict, f, indent=4)

    with (coint_dir / 'pvalue.json').open('w') as f:
        json.dump(scores_dict, f, indent=4)

    # with open('pvalue.json', 'w') as f:
    #     json.dump(pvalue_dict, f, indent=4)

    with (coint_dir / 'pairs.txt').open('w') as f:
        for name1, name2 in pairs:
            f.write('{},{}\n'.format(name1, name2))

    # with open('pairs.txt', 'w') as f:
    #     for name1, name2 in pairs:
    #         f.write('{},{}\n'.format(name1, name2))

    df_scores = dataframe_from_coint_dict(scores_dict)
    df_pvalue = dataframe_from_coint_dict(pvalue_dict)

    return df_scores, df_pvalue, pairs


def demo_code():
    """
    Basically this function creates two artificial stocks which are cointegrated and plots them so we
    can visualize the concepts.

    :return: None
    """
    # Generate daily returns
    Xreturns = np.random.normal(0, 1, 100)
    # sum up and shift the prices up
    something = np.cumsum(Xreturns)
    X = pd.Series(something, name='X') + 50
    X.plot(figsize=(15, 7))
    plt.show()
    # add some noise and create a second (cointegrated) plot
    noise = np.random.normal(0, 1, 100)
    Y = X + 5 + noise
    Y.name = 'Y'
    pd.concat([X, Y], axis=1).plot(figsize=(15, 7))
    plt.show()

    """## Illustrating Cointegration
    Let's get this right off the bat. Cointegration is NOT the same thing as correlation! Correlation means that the two variables are interdependent. If you studied statistics, you'll know that correlation is simply the covariance of the two variables normalized by their standard deviations.
    Cointegration is slightly different. It means that the ratio between two series will vary around a mean. So a linear combination like:
    *Y = αX + e*
    would be a stationary time series.
    Now what is a stationary time series? In simple terms, it's when a time series varies around a mean and the variance also varies around a mean. What matters most to US is that we know that if a series looks like its diverging and getting really high or low, we know that it will eventually revert back.
    I hope I haven't confused you too much.
    """
    ratios = Y / X
    ratios.plot(figsize=(15, 7))
    plt.axhline((Y / X).mean(), color='red', linestyle='--')
    plt.xlabel('Time')
    plt.legend(['Price Ratio', 'Mean'])
    plt.show()

    """Here is a plot of the ratio between the two two series. Notice how it tends to revert back to the mean? This is a clear sign of cointegration.
    ## Cointegration Test
    You now know what it means for two stocks to be cointegrated, but how do we actually quantify and test for cointegration?
    The module statsmodels has a good cointegration test that outputs a t-score and a p-value. It's a lot of statistical mumbo-jumbo that shows us the probability that we get a certain value given the distribution. In the end, we want to see a low p-value, ideally less than 5%, to give us a clear indicator that the pair of stocks are very likely to be cointegrated.
    """
    score, pvalue, _ = coint(X, Y)
    print(pvalue)


def update_coint_dict(coint_dict, value, name1, name2):
    """
    Updates the coint nested dictionary

    :param coint_dict:
    :param value:
    :param name1:
    :param name2:
    :return:
    """
    # place the value within the name1 -> name2 nested dict
    first_scores_dict: Dict = coint_dict.get(name1, {})
    first_scores_dict[name2] = value
    coint_dict[name1] = first_scores_dict
    # place the value within the name2 -> name1 nested dict
    second_scores_dict: Dict = coint_dict.get(name2, {})
    second_scores_dict[name1] = value
    coint_dict[name2] = second_scores_dict


def plot_tri_heatmap(dataframe: pd.DataFrame, title: str = '', save_dir: Path = None):
    """
    Plot a triangular heatmap from dataframe

    :param dataframe: pandas DataFrame
    :param title: title of the plot
    :return: None
    """

    # mask the top-right half
    mask = np.zeros_like(dataframe)
    mask[np.triu_indices_from(mask)] = True

    # create the actual heatmap
    # ax = plt.axes()
    fig, ax = plt.subplots(1, 1)

    sns.heatmap(dataframe, mask=mask, ax=ax)
    ax.set_title(title)

    # save plot as PNG or display when script is run
    if save_dir is not None:
        save_file: Path = save_dir / title
        fig.savefig(save_file)
    else:
        plt.show()


def plot_pair_ratio(dataframe: pd.DataFrame, pair: Tuple[str, str], save_dir: Path = None):
    """
    Given a dataframe and a pair of two columns, plot the ratio

    :param dataframe:
    :param pair: tuple or list containing two columns (stocks)
    :param save_dir: (optional) path to folder where to save plot files. \
    If not specified, plots will be rendered instead.
    :return: None
    """
    name1 = pair[0]
    name2 = pair[1]

    df_ratios = get_ratios_dataframe(dataframe, name1, name2)

    fig, ax = plt.subplots(3, 1, sharex='all', sharey='none')
    fig: Figure = fig       #: Re-Declared for type casting
    ax: Tuple[Axes] = ax    #: Re-Declared for type casting

    # set fig size
    fig.set_size_inches(30, 15.5)

    title_name: str = '{} vs {}'.format(name1, name2)
    fig.suptitle(title_name)

    # ratios plot
    plot_lineplt(
        df_ratios, y=['ratios', 'mean', '+1_std', '-1_std', 'mavg5', 'mavg60'],
        x_title='date', y_title='ratio', title='ratio against date', ax=ax[0])
    ax[0].plot(df_ratios['zscore_buy'], linestyle='None', marker='^', color='g')
    ax[0].plot(df_ratios['zscore_sell'], linestyle='None', marker='^', color='r')

    # plot the standardized z-score
    plot_lineplt(
        df_ratios, y=['zscore_60_5', 'zscore_+1', 'zscore_-1', 'zscore_0'],
        x_title='date', y_title='ratio', title='mean reversion', ax=ax[1])

    # plot the actual stocks
    plot_lineplt(
        df_ratios, y=[name1, name2],
        x_title='date', y_title='ratio', title='mean reversion', ax=ax[2])
    ax[2].plot(df_ratios['name1_buy'], linestyle='None', marker='^', color='g')
    ax[2].plot(df_ratios['name1_sell'], linestyle='None', marker='^', color='r')
    ax[2].plot(df_ratios['name2_buy'], linestyle='None', marker='^', color='g')
    ax[2].plot(df_ratios['name2_sell'], linestyle='None', marker='^', color='r')

    if save_dir is not None:
        save_dir.mkdir(exist_ok=True, parents=True)
        save_file = save_dir / title_name
        fig.savefig(save_file)

    else:
        fig.show()


def plot_lineplt(
        dataframe, y: List = None,
        x_title: str = 'x_title', y_title: str = 'y_title',
        title: str = 'title', ax: Axes = None):
    """
    Given a dataframe, list of columns, and x and y titles, this will plot a lineplot
    of all columns against the dataframe index.

    :param dataframe: pd.DataFrame
    :param y: list of columns names to plot against the index
    :param x_title: name of x axis title
    :param y_title: name of y axis title
    :param title: title of the plot
    :param ax: matplotlib ax for multiple plots together
    :return: None
    """

    # set x_ticks
    df_indices: np.ndarray = dataframe.index.values.reshape(-1, )
    skip_val: int = int(df_indices.shape[0] / 10)     # sample n number of x ticks so they are visible
    x_tick_labels = df_indices[::skip_val]

    # plot the line on the axis
    if y is None:
        y = list(dataframe.columns)

    for column in y:
        sns.lineplot(x=dataframe.index, y=column, data=dataframe, ax=ax)

    # set title and axes labels
    ax.set_title(title)
    ax.set_xlabel(x_title)
    ax.set_ylabel(y_title)

    # set the x-ticks
    ax.set_xticks(x_tick_labels)
    for tick in ax.get_xticklabels():
        tick: Text = tick  #: redeclaration of the variable so we can type check
        tick.set_rotation('45')
        tick.set_horizontalalignment('right')

    ax.legend(y)


def plot_pair_vs(dataframe: pd.DataFrame, pair: Tuple[str, str]):
    """
    Given a dataframe and a pair of two columns, plot the ratio

    :param dataframe: pandas DataFrame
    :param pair: tuple or list containing two columns (stocks)
    :return: None
    """
    name1 = pair[0]
    name2 = pair[1]

    ax = ax = plt.axes()
    sns.lineplot(x=name1, y=name2, data=dataframe, ax=ax)
    ax.set_title('{} against {}'.format(name2, name1))
    plt.show()


# Trade using a simple strategy
def trade(S1, S2, window1, window2):
    """
    Trade using a simple strategy.

    :param S1: pd.Series for stock 1
    :param S2: pd.Series for stock 2
    :param window1: (int) period of time to calculate rolling ma1
    :param window2: (int) period of time to calculate rolling ma2 and std
    :return: (float) money earned
    """
    # If window length is 0, algorithm doesn't make sense, so exit
    if (window1 == 0) or (window2 == 0):
        return 0

    # Compute rolling mean and rolling standard deviation
    ratios = S1 / S2
    ma1 = ratios.rolling(window=window1, center=False).mean()
    ma2 = ratios.rolling(window=window2, center=False).mean()
    std = ratios.rolling(window=window2, center=False).std()
    zscore = (ma1 - ma2) / std

    # Simulate trading, start with no money and no positions
    money, countS1, countS2 = 0, 0, 0

    for i in range(len(ratios)):

        # Sell short if the z-score is > 1
        if zscore[i] < -1:
            money += S1[i] - S2[i] * ratios[i]
            countS1 -= 1
            countS2 += ratios[i]

        # Buy long if the z-score is < -1
        elif zscore[i] > 1:
            money -= S1[i] - S2[i] * ratios[i]
            countS1 += 1
            countS2 -= ratios[i]

        # Clear positions if the z-score between -.5 and .5
        elif abs(zscore[i]) < 0.75:
            money += S1[i] * countS1 + S2[i] * countS2
            countS1 = 0
            countS2 = 0

    return money
