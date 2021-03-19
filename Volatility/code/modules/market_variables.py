import os
import pandas as pd
import numpy as np
from pathlib import Path
pd.set_option('mode.chained_assignment', None)


# load all cryptocurrencies's data
def market_variables(path, filenames):
    par = (Path(path)).parent
    mm = par / 'weighted_avg_proxies_vol'
    os.chdir(str(mm))

    ###################################################################################################################
    # PREPARE DATA
    ####################################################################################################################

    ff = filenames.copy()
    ff.sort()
    BCH = pd.read_csv(ff[0], sep=',', parse_dates=['date'], encoding='latin_1')
    BTC = pd.read_csv(ff[1], sep=',', parse_dates=['date'], encoding='latin_1')
    EOS = pd.read_csv(ff[2], sep=',', parse_dates=['date'], encoding='latin_1')
    ETH = pd.read_csv(ff[3], sep=',', parse_dates=['date'], encoding='latin_1')
    XRP = pd.read_csv(ff[4], sep=',', parse_dates=['date'], encoding='latin_1')

    # match times across CCies, so that for each ccy we have the same time indices
    # this is needed in order to compute complementary "market" proxies
    intersection_dates = list(set.intersection(set(BCH.date), set(BTC.date), set(EOS.date), set(ETH.date), set(XRP.date)))
    # we subset our ccies based on this intersection of dates and we reset the index of the vector
    # for easier vector operations
    BCH = BCH.loc[BCH.date.isin(intersection_dates), :].reset_index()
    BTC = BTC.loc[BTC.date.isin(intersection_dates), :].reset_index()
    EOS = EOS.loc[EOS.date.isin(intersection_dates), :].reset_index()
    ETH = ETH.loc[ETH.date.isin(intersection_dates), :].reset_index()
    XRP = XRP.loc[XRP.date.isin(intersection_dates), :].reset_index()

    # weights for Market-wide variable such as Returns and Liquidity-proxies
    # equal weights
    BCH["w_idx"] = 0.2
    BTC["w_idx"] = 0.2
    EOS["w_idx"] = 0.2
    ETH["w_idx"] = 0.2
    XRP["w_idx"] = 0.2

    ###################################################################################################################
    # COMPUTE -- Market Proxies --
    ####################################################################################################################

    # we build a function that computes the Market Proxies (taking the originating ccy out of the equation each time)
    def complementaryMARKETproxy(proxy_name, crypto_name):
        C = {"BCH": BCH, "BTC": BTC, "EOS": EOS, "ETH": ETH, "XRP": XRP}
        L = pd.Series(np.zeros(BCH.shape[0]))
        for key, val in C.items():
            if key != crypto_name:
                L += val[proxy_name] * val["w_idx"]
        return L

    # extend original dataframes witth their corresponding MARKET WIDE proxies
    proxies = ["EWPQS", "EWDEPTH", "SWPQS", "SWPES", "SWPTS", "TWPQS", "return"]

    for p in proxies:
        fstring = f"MKT_{p}"
        BCH[fstring] = complementaryMARKETproxy(p, "BCH")
        BTC[fstring] = complementaryMARKETproxy(p, "BTC")
        EOS[fstring] = complementaryMARKETproxy(p, "EOS")
        ETH[fstring] = complementaryMARKETproxy(p, "ETH")
        XRP[fstring] = complementaryMARKETproxy(p, "XRP")

    ###################################################################################################################
    # EXTEND -- 1-Lag, 1-Lead Proxies --
    ####################################################################################################################

    for p in ["MKT_EWPQS", "MKT_EWDEPTH", "MKT_SWPQS", "MKT_SWPES", "MKT_SWPTS", "MKT_TWPQS", "MKT_return"]:
        BCH[p + "_LAG1"] = BCH[p].shift(periods=1, axis=0)
        BTC[p + "_LAG1"] = BTC[p].shift(periods=1, axis=0)
        EOS[p + "_LAG1"] = EOS[p].shift(periods=1, axis=0)
        ETH[p + "_LAG1"] = ETH[p].shift(periods=1, axis=0)
        XRP[p + "_LAG1"] = XRP[p].shift(periods=1, axis=0)
        BCH[p + "_LEAD1"] = BCH[p].shift(periods=-1, axis=0)
        BTC[p + "_LEAD1"] = BTC[p].shift(periods=-1, axis=0)
        EOS[p + "_LEAD1"] = EOS[p].shift(periods=-1, axis=0)
        ETH[p + "_LEAD1"] = ETH[p].shift(periods=-1, axis=0)
        XRP[p + "_LEAD1"] = XRP[p].shift(periods=-1, axis=0)

    # remove first and last observation for each CCY
        BCH = BCH[1:-1]
        BTC = BTC[1:-1]
        EOS = EOS[1:-1]
        ETH = ETH[1:-1]
        XRP = XRP[1:-1]

    base = par / 'base_for_regression'
    base.mkdir(parents=True, exist_ok=True)
    os.chdir(str(base))

    BCH.to_csv("bitfinex_bchusd.csv", index=False, encoding='latin_1')
    BTC.to_csv("bitfinex_btcusd.csv", index=False, encoding='latin_1')
    EOS.to_csv("bitfinex_eosusd.csv", index=False, encoding='latin_1')
    ETH.to_csv("bitfinex_ethusd.csv", index=False, encoding='latin_1')
    XRP.to_csv("bitfinex_xrpusd.csv", index=False, encoding='latin_1')
    print("***---------------------------------")
    print("***     Market-wide proxies have been generated")
    print("***     under './base_for_regression'")
    print("***---------------------------------")
