import os
import pandas as pd
import numpy as np
from pathlib import Path
import datetime as dt
pd.set_option('mode.chained_assignment', None)


def preprocess_ob(path, filename):
    xpath = Path(path) / 'csv'
    os.chdir(xpath)

    # load
    cols = ['datetime', 'PB01', 'QB01', 'PA01', 'QA01']
    ob = pd.read_csv(filename, usecols=cols, encoding='latin_1')
    ob.datetime = ob.datetime.str.slice(0, 19)
    ob.datetime = pd.to_datetime(ob.datetime, format='%Y-%m-%d %H:%M:%S')
    # clean and reduce
    ob = ob.loc[(ob['PB01'] >= 0) & (ob['QB01'] >= 0), ]
    ob = ob.loc[(ob['PA01'] >= 0) & (ob['QA01'] >= 0), ]
    # time scope reduction
    mask = (ob['datetime'] >= '2017-08-11') & (ob['datetime'] <= '2018-07-21')
    ob = ob.loc[mask]
    # sort by datetime and reset index
    ob.sort_values(by=['datetime'], inplace=True, ascending=True)
    ob.reset_index(inplace=True, drop=True)
    return ob


def preprocess_trades(path, filename):
    xpath = Path(path) / 'csv'
    os.chdir(xpath)
    # load
    cols = ['datetime', 'price', 'amount', 'sell']
    trades = pd.read_csv(filename, usecols=cols, dtype={'sell': 'boolean'}, encoding='latin_1')
    trades.datetime = trades.datetime.str.slice(0, 19)
    trades.datetime = pd.to_datetime(trades.datetime, format='%Y-%m-%d %H:%M:%S')
    # clean and reduce
    trades = trades.loc[(trades['price'] >= 0) & (trades['amount'] >= 0), ]
    # time scope reduction
    mask = (trades['datetime'] >= '2017-08-11') & (trades['datetime'] <= '2018-07-21')
    trades = trades.loc[mask]
    #  change values in sell column, True becomes -1 and False becomes +1
    trades.sell = trades.sell.map({True: -1, False: 1})
    trades.rename(columns={'sell': 'dir'}, inplace=True)
    # sort by datetime and reset index
    trades.sort_values(by=['datetime'], inplace=True, ascending=True)
    trades.reset_index(inplace=True, drop=True)
    return trades


def preprocess_trades_ob(path, filename):
    # we build the path for trades
    # xpath = Path(path) / 'csv'
    # corresponding path for ob
    ypath = complementary_dir(path)
    # os.chdir(xpath)

    trades = preprocess_trades(path, filename)
    # load corresponding OB file so that we can match trades datetimes to OB datetimes
    ob = preprocess_ob(ypath, filename)
    trades = correspond_datetimes(trades, ob)

    # compute duration of quotes
    trades, ob = duration_of_quotes(trades, ob)

    # compute ex ante liquidity proxies
    ob = ex_ante_liq_proxy(ob)

    # merge trades and ob
    merged = merge_traded_ob(trades, ob)

    # compute ex post liquidity proxies
    merged = ex_post_liq_proxy(merged)

    # smoothen the series: take the median record over 5 minute intervals
    medianized = bundle_time_intervals(5, merged)
    print(medianized.head())
    # save output to "/medianized" folder
    save_medianized(medianized, path, filename)


def complementary_dir(path):
    isTrade = 'trades' in path
    isOB = 'ob' in path
    tmp = (Path(path)).parent
    if isTrade:
        ypath = tmp / 'ob'
    if isOB:
        ypath = tmp / 'trades'
    return str(ypath)


def correspond_datetimes(td, obook):
    # Match datetimes
    tt = np.array(td.datetime)
    ot = np.array(obook.datetime)
    nt = np.zeros(tt.shape, dtype='datetime64[ns]')
    cursor = 0
    for i, t in enumerate(tt):
        if i % 1000000 == 0:
            new_i = i / 1000000
            print(f"{new_i} million trades processed")
        for j, o in enumerate(ot[cursor:]):
            if t < o:
                nt[i] = ot[cursor + j - 1]
                cursor += j
                break
            elif t == o:
                nt[i] = ot[cursor + j]
                cursor += j
                break
    print()
    nt = pd.Series(nt)
    td["corresp_OB_datetime"] = nt

    # remove NULL-datetime records (no corresponding OB datetime)
    tra = td.loc[td.corresp_OB_datetime != np.datetime64('1970-01-01 00:00:00'), :]
    return tra


def duration_of_quotes(trades, ob):
    # DURATION of quotes
    # 1) we start by computing the delta_times between each consecutive pair of quotes
    null_td = pd.Timedelta(seconds=0)
    ob["delta_time"] = (ob.datetime.diff().fillna(null_td)).apply(pd.Timedelta.total_seconds)

    # since we compute diff --> delete first datetime record from ob  and its linked trades
    # enregistrer la date qui pose probleme d abord
    date_cassepied = ob.iloc[0].datetime

    # retirer la premiere observation de TRADES car delta_time non-définie pour le premier element
    ob = ob.iloc[1:]
    ob.reset_index(drop=True, inplace=True)

    # du coup retirer aussi les equivalents dans TRADES, en utilisant la date sauvegardée 'date_cassepied'
    trades = trades.loc[trades.corresp_OB_datetime != np.datetime64(date_cassepied), :]
    trades.reset_index(drop=True, inplace=True)

    # retirer tous les trades qui n'ont pas un corresp ob time inferieur ou egal à la minute
    L = list((ob.loc[ob.delta_time > 60, "datetime"]))
    idx = trades.loc[trades.corresp_OB_datetime.isin(L)].index
    trades.drop(idx, inplace=True)
    trades.reset_index(drop=True, inplace=True)

    # 2) We check whether the quote has changed through time and we compute the DURATION
    # for this we use our previously computed delta times & use numpy for faster execution
    A = np.array(ob.PA01)
    B = np.array(ob.PB01)
    T = np.array(ob.datetime)
    D = np.array(ob.delta_time)
    TW = np.zeros(T.shape)
    MAX_DELTA_TIME = 60  # max 60 seconds between consecutive order book times

    for index, (a, b, t, d) in enumerate(zip(A, B, T, D)):
        if index == 0:
            prev_PB = b
            prev_PA = a
            tw = 0
            block_count = 0
        else:
            if (a != prev_PA or b != prev_PB or index == len(T) - 1 or d > MAX_DELTA_TIME):
                if block_count > 1:
                    x = TW[index - 1]
                    TW[index - block_count:index] = x
                tw = 0
                block_count = 0
                prev_PB = b
                prev_PA = a
        if d <= MAX_DELTA_TIME:
            tw += d
        else:  # d > MAX_DELTA_TIME
            tw = MAX_DELTA_TIME
        block_count += 1
        TW[index] = tw

    ob.insert(len(ob.columns), 'tw', TW)
    ob = ob[["datetime", "PB01", "QB01", "PA01", "QA01", "tw"]]
    return trades, ob


def ex_ante_liq_proxy(ob):
    # midpoint quote price
    ob["midpoint"] = (ob["PA01"] + ob["PB01"]) / 2
    # depth
    ob["DEPTH"] = (ob["QA01"] + ob["QB01"]) / 2
    # PQS"  Proportional (percent) Quoted Spread "
    ob["PQS"] = (ob["PA01"] - ob["PB01"]) / ob["midpoint"]
    return ob


def merge_traded_ob(trades, ob):
    # remove duplicates on datetimes
    ob = ob.groupby(["datetime"], as_index=False).median()

    # fusion des tables
    merged = pd.merge(left=trades, right=ob, how="left", left_on="corresp_OB_datetime", right_on="datetime")
    newcols = ['datetime_x', 'price', 'amount', 'dir', 'corresp_OB_datetime', 'PB01',
               'QB01', 'PA01', 'QA01', 'midpoint', 'tw', 'DEPTH', 'PQS']
    merged = merged[newcols]
    merged = merged.rename(columns={"datetime_x": "datetime"})
    return merged


def ex_post_liq_proxy(merged):
    # Proportional (percent) Effective Spread
    merged["PES"] = (2 * merged.dir * (merged.price - merged.midpoint)) / merged.midpoint

    # Proportional (percent) Trade Spread
    merged["PTS"] = merged.PES / 2
    return merged


def bundle_time_intervals(time_interval, df):
    print("     The DatFrame is to be reformated taking the medians")
    print(f"    over {time_interval} - minute intervals")
    # we observed that we have 1 ob record every minutes, and multiple trades corresponding
    # to this ob record
    # by making X-min intervals we shall have intervals of size X (more or less)
    # we make sure to have at least 3 after this

    def X_MinClassifier(instant):
        discard = dt.timedelta(minutes=instant.minute % time_interval, seconds=instant.second)
        instant -= discard
        if discard <= dt.timedelta(minutes=time_interval):
            instant += dt.timedelta(minutes=time_interval)
        return instant

    # proceed to bundle by 5min
    # example: 00:05:00 interval will hold contain all trades from 00:00:00 up to 00:05:00

    # remove interval with less than 3 trades
    df["interval"] = df.corresp_OB_datetime.apply(X_MinClassifier)
    a = pd.DataFrame(df.groupby(["interval"]).count()["price"])
    a.reset_index(level=0, inplace=True)
    S = list(a[a.price < 3].interval)
    df = df.loc[~df.interval.isin(S)]

    medianized_proxys = df.groupby(["interval"]).median()
    print("Medianization status:  <<< Done >>>")
    print()
    return medianized_proxys


def save_medianized(df, path, filename):
    par = (Path(path)).parent
    mm = par / 'medianized'
    mm.mkdir(parents=True, exist_ok=True)
    os.chdir(str(mm))
    df.to_csv(filename, index=True, encoding='latin_1')
    print(f"File <<< {filename} >>> has been saved in './medianized' folder ")
    print()
