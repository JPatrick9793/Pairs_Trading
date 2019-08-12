# -*- coding: utf-8 -*-
import json
from datetime import datetime
from pathlib import Path
from typing import Tuple, Dict, List, Any

import numpy as np
import pandas as pd
from tqdm import tqdm

multi_index_names = ['year', 'month', 'day', 'hour', 'min', 'sec']  #: List of multi-index names


def collapse_multiindex_date(time_tuple: Tuple):
    """ Function to convert the multi-index dataframe into single index """
    return datetime(*time_tuple[0:6]).strftime("%Y-%m-%d %H:%M:%S")


def expand_multiindex_date(timestamp: pd.datetime):
    """ Function to convert the single index dataframe into multi-index """
    tm_year, tm_mon, tm_mday, tm_hour, tm_min, tm_sec, *_ = timestamp.timetuple()
    return tuple([tm_year, tm_mon, tm_mday, tm_hour, tm_min, tm_sec])


def load_prices(prices_dir: Path) -> pd.DataFrame:
    """
    Given the path to a directory containing CSV files of stock closing data this function will
    merge all those CSVs together into giant dataframe.

    :param prices_dir: Path object pointing to PRICE directory
    :return: pd.DataFrame
    """

    # csvs will be merged on this column
    merge_column: str = 'date'

    # iterate through directory
    for i, file_path in tqdm(enumerate([f for f in prices_dir.iterdir() if f.is_file()]), 'Iterating price CSVs'):

        stock_ticker: str = file_path.name.split('.')[0]
        rename_columns: Dict = {'close': '{}'.format(stock_ticker)}

        # if first file, create base DataFrame for future merges
        if i == 0:

            df = pd.read_csv(str(file_path))
            df.rename(columns=rename_columns, inplace=True)

            continue

        else:

            sub_df = pd.read_csv(str(file_path))
            sub_df.rename(columns=rename_columns, inplace=True)
            df = df.merge(sub_df, sort=False, on=merge_column)

    # set the index to date so it doesnt get included in the downstream processing steps
    df.set_index('date', inplace=True)

    df.to_hdf('PRICE.h5', key='df')

    return df


def dataframe_from_coint_dict(coint_dict: Dict) -> pd.DataFrame:
    """
    Given a nested dictionary, create a pandas dataframe

    :param coint_dict: Nested dictionary containing either pvalues or t-test scores
    :return: pandas DataFrame
    """
    df = pd.DataFrame(coint_dict).sort_index()
    df = df.reindex(columns=df.index.values)
    return df


def get_series_zscore(series: pd.Series) -> pd.Series:
    """
    Returns a pandas series containing zscores
    :param series: pandas series object (numeric type)
    :return: pandas series
    """
    return (series - series.mean()) / series.std()


def get_ratios_dataframe(dataframe: pd.DataFrame, name1: str, name2: str) -> pd.DataFrame:
    """
    Given a dataframe and two column names, this function will return another dataframe which contains
    the same index, and two columns:
        - 'ratios' name1/name2
        - 'mean' mean of ratios column (same value in every row)
        - '+1_std' one standard deviation above mean (same value in every row)
        - '-1_std' one standard deviation below mean (same value in every row)

    :param dataframe: pd.DataFrame
    :param name1: name of first column
    :param name2: name of second column
    :return:
    """
    # calculate name1/name2 ratio
    df_ratios: pd.DataFrame = (dataframe[name1] / dataframe[name2]).to_frame(name='ratios')
    # calculate the mean of the ratios
    df_ratios['mean'] = df_ratios['ratios'].mean()
    # add ± 1 standard devation
    std_series = df_ratios['ratios'].std()
    df_ratios['+1_std'] = df_ratios['mean'] + std_series
    df_ratios['-1_std'] = df_ratios['mean'] - std_series

    # add the original columns back in the dataframe
    df_ratios[name1] = dataframe[name1]
    df_ratios[name2] = dataframe[name2]

    # get rolling mean and std stats
    df_ratios['mavg5'] = df_ratios['ratios'].rolling(window=5, center=False).mean().fillna(method='backfill')
    df_ratios['mavg60'] = df_ratios['ratios'].rolling(window=60, center=False).mean().fillna(method='backfill')
    df_ratios['std_60'] = df_ratios['ratios'].rolling(window=60, center=False).std().fillna(method='backfill')

    # using the rolling stats, we can calculate the rolling z-score
    df_ratios['zscore_60_5'] = ((df_ratios['mavg5']-df_ratios['mavg60'])/df_ratios['std_60']).fillna(method='backfill')
    df_ratios['zscore_+1'] = +1
    df_ratios['zscore_-1'] = -1
    df_ratios['zscore_0'] = 0

    # find buy/sell signals using that rolling z-score for the ratio plot
    df_ratios['zscore_buy'] = df_ratios.apply(create_apply_signal(value_column='ratios', which='buy'), axis=1)
    df_ratios['zscore_sell'] = df_ratios.apply(create_apply_signal(value_column='ratios', which='sell'), axis=1)

    # # When you buy the ratio, you buy stock S1 and sell S2
    # buyR[buy != 0] = S1[buy != 0]
    # sellR[buy != 0] = S2[buy != 0]
    # # When you sell the ratio, you sell stock S1 and buy S2
    # buyR[sell != 0] = S2[sell != 0]
    # sellR[sell != 0] = S1[sell != 0]

    # find buy/sell signals for original columns
    # ------------------------------------------
    # When you buy the ratio, you buy stock S1 and sell S2
    df_ratios['name1_buy'] = df_ratios.apply(create_apply_signal(value_column=name1, which='buy'), axis=1)
    df_ratios['name2_sell'] = df_ratios.apply(create_apply_signal(value_column=name2, which='buy'), axis=1)
    # When you sell the ratio, you sell stock S1 and buy S2
    df_ratios['name1_sell'] = df_ratios.apply(create_apply_signal(value_column=name1, which='sell'), axis=1)
    df_ratios['name2_buy'] = df_ratios.apply(create_apply_signal(value_column=name2, which='sell'), axis=1)

    # return the dataframe
    return df_ratios


def create_apply_signal(value_column: str = 'ratios', which: str = 'buy'):
    """
    Function which returns another function that can be passed into the pandas.DataFrame.apply(axis=1) method.

    :param value_column: Name of DataFrame column whose values will be used when the signal criteria is met
    :param which: Either 'sell' or 'buy' depending on the signal you are looking for
    :return: Callable Function
    """

    # make sure the types are checked
    assert isinstance(value_column, str), TypeError('\'value_column\' must be type \'str\'')
    assert isinstance(which, str), TypeError('\'which\' must be type \'str\'')

    # define the buy function
    def apply_buy_signal(row: Dict):
        zscore_60_5 = row['zscore_60_5']
        ratio = row[value_column]
        if zscore_60_5 < -1: return ratio
        else: return np.nan

    # define the sell function
    def apply_sell_signal(row: Dict):
        zscore_60_5 = row['zscore_60_5']
        ratio = row[value_column]
        if zscore_60_5 > +1: return ratio
        else: return np.nan

    # check that 'which' has a proper value
    if which.lower() == 'buy':
        return apply_buy_signal
    elif which.lower() == 'sell':
        return apply_sell_signal
    else:
        raise ValueError('argument \'which\' must be either \'buy\' or \'sell\'.'
                         'Instead, received: {}'.format(which))


def load_coint_tests(coint_dir=Path('tmp')) -> Tuple[
    pd.DataFrame, pd.DataFrame, np.ndarray]:
    """
    This function will load the results from the do_coint_test() function.

    :param coint_dir: Path object pointing to directory which contains 'scores.json', 'pvalue.json', and 'pairs.txt'
    :return: (df_scores, df_pvalue, pairs)
    """

    scores_file = coint_dir / 'scores.json'
    pvalue_file = coint_dir / 'pvalue.json'
    pairs_file = coint_dir / 'pairs.txt'

    scores_dict = json.load(scores_file.open('r'))
    df_scores = dataframe_from_coint_dict(scores_dict)
    df_scores.index.astype(pd.datetime, copy=False)

    pvalue_dict = json.load(pvalue_file.open('r'))
    df_pvalue = dataframe_from_coint_dict(pvalue_dict)
    df_scores.index.astype(pd.datetime, copy=False)

    pairs = np.loadtxt(str(pairs_file.resolve()), delimiter=',', dtype=str, ndmin=2)

    return df_scores, df_pvalue, pairs


def aggregate_dataframe(dataframe: pd.DataFrame, groupby: str = 'day', aggby: str = 'mean'):
    # convert dataframe index to datetime
    df2 = dataframe.set_index(pd.DatetimeIndex(dataframe.index))

    # split single index into multiindex
    df2.index = pd.MultiIndex.from_tuples(
        tuples=[expand_multiindex_date(k) for k, v in df2.iterrows()],
        names=multi_index_names)

    # group by something
    df2 = df2.groupby(
        list_slice_by_val(x=multi_index_names, val=groupby)
    ).agg([aggby])

    # flatten the dataframe columns
    df2.columns = df2.columns.get_level_values(0)

    # flatten the dataframe multi-index
    df2.index = df2.index.map(collapse_multiindex_date)

    return df2


def list_slice_by_val(x: List, val: Any):
    """ Generic utility function which slices a list (or array) by value, instead of index """
    return x[:x.index(val) + 1]
