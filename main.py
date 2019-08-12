# -*- coding: utf-8 -*-
import json
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm
from sklearn.model_selection import train_test_split

from routines import dataframes, utils

np.random.seed(107)


if __name__ == "__main__":

    # NOTE switch the lines if you want to load from file instead of running from scratch
    # path_to_prices: Path = Path('PRICE').resolve()
    # df: pd.DataFrame = dataframes. load_prices(path_to_prices)
    df: pd.DataFrame = pd.read_hdf('PRICE.h5', key='df')   # Results have been stored in h5 file for speed

    # group the dataframe and aggregate
    df_grouped = dataframes.aggregate_dataframe(
        dataframe=df, groupby='hour', aggby='mean')

    # split the DataFrame into inference and back-testing sets
    df_train, df_test = train_test_split(df_grouped, train_size=0.8, shuffle=False)

    # NOTE run this if you want to see the demo code from article
    # routines.demo_code()

    # TODO include option for Johansen test
    # NOTE switch the lines if you want to load from file instead of running from scratch
    coint_dir = Path('hour_mean')
    # df_scores, df_pvalue, pairs = utils.do_coint_tests(dataframe=df_train, pvalue_threshold=0.05, coint_dir=coint_dir)
    df_scores, df_pvalue, pairs = dataframes.load_coint_tests(coint_dir=coint_dir)

    utils.plot_tri_heatmap(df_scores, title='t-test scores', save_dir=coint_dir)
    utils.plot_tri_heatmap(df_pvalue, title='p-values scores', save_dir=coint_dir)

    # dictionary for storing back-testing results
    back_testing_dictionary = {}
    # sub-folder to hold the plots
    save_dir: Path = coint_dir / 'plots'
    # iterate through the pairs to backtest and plot
    for i, pair in tqdm(enumerate(pairs), desc='testing pairs...'):
        name1 = pair[0]
        name2 = pair[1]

        # pot the graphs
        utils.plot_pair_ratio(dataframe=df_test, pair=pair, save_dir=save_dir)

        # perform back-testing
        money = utils.trade(
            df_test[name1],
            df_test[name2],
            60, 5)

        # extract other metrics from this pair of stocks
        money = round(money, 3)
        pvalue = df_pvalue[name1][name2]
        tscore = df_scores[name1][name2]
        back_testing_dictionary['{}_{}'.format(name1, name2)] = {
            'money': money,
            'pvalue': pvalue,
            'tscore': tscore}

        # print('{:5} and {:5}: ${}'.format(name1, name2, money))

    # save the back-testing results
    with (coint_dir / 'backtesting.json').open('w') as f:
        json.dump(back_testing_dictionary, f, indent=4)
