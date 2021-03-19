import os
import pandas as pd
import numpy as np
from pathlib import Path
pd.set_option('mode.chained_assignment', None)


def weihted_averages(path, filename):
    """
    compute single proxies for all 15-min intervals and for all CCIES
    """

    par = (Path(path)).parent
    mm = par / 'medianized'
    os.chdir(str(mm))
    # LOADING DATA
    med = pd.read_csv(filename, parse_dates=["interval"], encoding='latin_1')

    # extend columns announcing the year, month, day, hout, minute
    extend_by_split_datetime(med)

    # COMPUTE RETURN ,SQUARED RETURNS AND WEIGHTED AVERAGES OF PROXIES
    WProx = get_weighted_avg_proxies(med)

    # load Vcrix data and extend WProx with volatility flag
    vcrix = load_vcxrix(path)
    WProx = extend_wavg_with_vcrix(path, WProx, vcrix)

    # save file
    ww = par / 'weighted_avg_proxies_vol'
    ww.mkdir(parents=True, exist_ok=True)
    os.chdir(str(ww))
    WProx.to_csv(filename, index=False, encoding='latin_1')
    print(f"File <<< {filename} >>> has been saved in './weighted_avg_proxies' folder ")


def extend_by_split_datetime(df):
    # buil columns announcing the year, month, day, hout, minute of the medianazed DataFrame
    Y = pd.DatetimeIndex(np.array(df.interval)).year
    M = pd.DatetimeIndex(np.array(df.interval)).month
    D = pd.DatetimeIndex(np.array(df.interval)).day
    H = pd.DatetimeIndex(np.array(df.interval)).hour
    T = pd.DatetimeIndex(np.array(df.interval)).minute
    G = np.zeros(T.shape)

    for index, (t, g) in enumerate(zip(T, G)):
        if t == 0 or t == 5 or t == 10:
            g = 15
        elif t == 15 or t == 20 or t == 25:
            g = 30
        elif t == 30 or t == 35 or t == 40:
            g = 45
        elif t == 45 or t == 50 or t == 55:
            g = 60
        G[index] = g

    df.insert(len(df.columns), 'group', G)
    df.insert(len(df.columns), 'hour', H)
    df.insert(len(df.columns), 'day', D)
    df.insert(len(df.columns), 'month', M)
    df.insert(len(df.columns), 'year', Y)


def get_weighted_avg_proxies(df):
    def ret(x):
        y = x[1:]
        return sum(y)
    # compute log returns over our base : 5 min intervals
    EW = df.groupby(["year", "month", "day", "hour", "group"]).mean()
    EW["return"] = (np.log(EW.price)).diff()
    EW.reset_index(drop=False, inplace=True)

    # compute Equally Weighted proxies ...
    EW = EW[["year", "month", "day", "hour", "group", "PQS", "DEPTH", "return"]]
    EW["V"] = np.power(EW["return"], 2)
    EW = EW.rename(columns={"PQS": "EWPQS", "DEPTH": "EWDEPTH"})

    # compute Size Weighted proxies ...
    def wm_pqs(x): return np.average(x, weights=df.loc[x.index, "DEPTH"])
    def wm_pes(x): return np.average(x, weights=df.loc[x.index, "amount"])
    def wm_pts(x): return np.average(x, weights=df.loc[x.index, "amount"])
    f = {'PQS': wm_pqs, 'PES': wm_pes, 'PTS': wm_pts}
    SW = df.groupby(["year", "month", "day", "hour", "group"]).agg(f)
    SW.reset_index(drop=False, inplace=True)
    SW = SW[["year", "month", "day", "hour", "group", "PQS", "PES", "PTS"]]
    SW = SW.rename(columns={"PQS": "SWPQS", "PES": "SWPES", "PTS": "SWPTS"})

    # Compute Time weighted proxies ...
    def wm_s(x): return np.average(x, weights=df.loc[x.index, "tw"])
    f = {'PQS': wm_s}
    TW = df.groupby(["year", "month", "day", "hour", "group"]).agg(f)
    TW.reset_index(drop=False, inplace=True)
    TW = TW[["year", "month", "day", "hour", "group", "PQS"]]
    TW = TW.rename(columns={"PQS": "TWPQS"})

    # merge everything
    WProx = pd.merge(EW, SW, how="left", on=["year", "month", "day", "hour", "group"]).merge(TW, how="left", on=["year", "month", "day", "hour", "group"])

    # remove first record since to return on that one
    WProx = WProx.iloc[1:]
    WProx.reset_index(drop=True, inplace=True)

    WProx.rename(columns={'group': 'minute'}, inplace=True)
    WProx["date"] = pd.to_datetime(WProx[['year', 'month', 'day', 'hour', 'minute']])

    return WProx


def load_vcxrix(path):
    par = (Path(path)).parent
    vv = par / 'vcrix'
    os.chdir(str(vv))
    # source <-- https://www.thecrix.de/
    vcrix = pd.read_csv('vcrix.csv', usecols=['date', 'vcrix'], sep=',', parse_dates=['date'])

    # put year, month and day info on separate columns
    Y = pd.DatetimeIndex(np.array(vcrix.date)).year
    M = pd.DatetimeIndex(np.array(vcrix.date)).month
    D = pd.DatetimeIndex(np.array(vcrix.date)).day
    vcrix.insert(len(vcrix.columns), 'day', D)
    vcrix.insert(len(vcrix.columns), 'month', M)
    vcrix.insert(len(vcrix.columns), 'year', Y)

    # set thresholds
    # lower bound :=  33% quantile of vcrix column
    lower_bound = vcrix.vcrix.quantile(0.33)
    # upper bound := 67% quantile ...
    upper_bound = vcrix.vcrix.quantile(0.66)

    # create and fill the volatily column
    vcrix.loc[(vcrix.vcrix < lower_bound), "volatility"] = "L"
    vcrix.loc[(lower_bound <= vcrix.vcrix) & (vcrix.vcrix <= upper_bound), "volatility"] = "M"
    vcrix.loc[(vcrix.vcrix > upper_bound), "volatility"] = "H"

    # make the 'volatiity' feature a categorical variable
    vcrix.volatility = vcrix.volatility.astype('category')
    vcrix.volatility = vcrix.volatility.cat.set_categories(new_categories=["L", "M", "H"], ordered=True)
    return vcrix


def extend_wavg_with_vcrix(path, Wpx, vcrix):
    # merge volatilty dataframe with our datframe of liquidity proxies
    vol_df = vcrix[["day", "month", "year", "volatility"]]
    proxies_and_vol = pd.merge(left=Wpx, right=vol_df, how='left', on=["year", "month", "day"])
    return proxies_and_vol
