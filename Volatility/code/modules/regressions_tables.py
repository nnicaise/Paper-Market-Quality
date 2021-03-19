import os
import pandas as pd
import numpy as np
from pathlib import Path
from scipy.stats import norm
import statsmodels.api as sm
import statsmodels.formula.api as smf
pd.set_option('mode.chained_assignment', None)


"""
            T A B L E S

"""


def agg_liq_measures(path):
    par = (Path(path)).parent
    mm = par / 'weighted_avg_proxies_vol'
    os.chdir(str(mm))

    cols = ['EWPQS', 'EWDEPTH', 'SWPQS', 'SWPTS', 'SWPES', 'TWPQS']
    df = pd.DataFrame(columns=cols)
    tc = [pd.read_csv(f, usecols=cols, encoding='latin_1') for f in os.listdir()]
    df = pd.concat(tc, ignore_index=True)
    mean_series = df.mean()
    median_series = df.median()
    std_series = df.std()
    ac1 = [df[c].autocorr(1) for c in df.columns]
    ix = df.columns.tolist()
    ac_series = pd.Series(data=ac1, index=ix)

    table2 = pd.concat([mean_series, median_series, std_series, ac_series], axis=1)
    table2.rename(columns={0: 'mean', 1: 'median', 2: 'std_dev', 3: 'ac(1)'}, inplace=True)
    fig_initiate()
    fig_title("<<< Aggregate Liquidity Measure -- Summary Statistics >>>")
    print(table2)
    fig_terminate()


def stdzed_liq_measures_per_vol_regime(path):
    par = (Path(path)).parent
    mm = par / 'weighted_avg_proxies_vol'
    os.chdir(str(mm))
    cols = ['hour', 'minute', 'EWPQS', 'EWDEPTH', 'SWPQS', 'SWPTS', 'SWPES', 'TWPQS', 'volatility']
    tc = [pd.read_csv(f, usecols=cols, encoding='latin_1') for f in os.listdir()]
    df = pd.concat(tc, ignore_index=True)
    # standardize data (each record standardized wrt its global interval mean and std dev)
    df_std = df.groupby(['hour', 'minute']).transform(lambda x: (x - x.mean()) / x.std())

    df_std['volatility'] = df['volatility']
    cols = ['EWPQS', 'EWDEPTH', 'SWPQS', 'SWPTS', 'SWPES', 'TWPQS', 'volatility']
    df_std = df_std[cols]

    # separate among H, L, MC
    new_cols = ['EWPQS', 'EWDEPTH', 'SWPQS', 'SWPTS', 'SWPES', 'TWPQS']
    high_vol_df = df_std[df_std.volatility == 'H']
    high_vol_df = high_vol_df[new_cols]
    medium_vol_df = df_std[df_std.volatility == 'M']
    medium_vol_df = medium_vol_df[new_cols]
    low_vol_df = df_std[df_std.volatility == 'L']
    low_vol_df = low_vol_df[new_cols]

    # high vol
    mean_series = high_vol_df.mean()
    median_series = high_vol_df.median()
    std_series = high_vol_df.std()
    ac1 = [high_vol_df[c].autocorr(1) for c in high_vol_df.columns]
    ix = high_vol_df.columns.tolist()
    ac_series = pd.Series(data=ac1, index=ix)

    table2 = pd.concat([mean_series, median_series, std_series, ac_series], axis=1)
    table2.rename(columns={0: 'mean', 1: 'median', 2: 'std_dev', 3: 'ac(1)'}, inplace=True)
    fig_initiate()
    fig_title("<<< Liquidity Measures in HIGH vol regime -- Summary Statistics (Standardized Data) >>>")
    print(table2)
    fig_terminate()

    # medium vol
    mean_series = medium_vol_df.mean()
    median_series = medium_vol_df.median()
    std_series = medium_vol_df.std()
    ac1 = [medium_vol_df[c].autocorr(1) for c in medium_vol_df.columns]
    ix = medium_vol_df.columns.tolist()
    ac_series = pd.Series(data=ac1, index=ix)

    table2 = pd.concat([mean_series, median_series, std_series, ac_series], axis=1)
    table2.rename(columns={0: 'mean', 1: 'median', 2: 'std_dev', 3: 'ac(1)'}, inplace=True)
    fig_initiate()
    fig_title("<<< Liquidity Measures in MEDIUM vol regime -- Summary Statistics (Standardized Data) >>>")
    print(table2)
    fig_terminate()

    # low vol
    mean_series = low_vol_df.mean()
    median_series = low_vol_df.median()
    std_series = low_vol_df.std()
    ac1 = [low_vol_df[c].autocorr(1) for c in low_vol_df.columns]
    ix = medium_vol_df.columns.tolist()
    ac_series = pd.Series(data=ac1, index=ix)

    table2 = pd.concat([mean_series, median_series, std_series, ac_series], axis=1)
    table2.rename(columns={0: 'mean', 1: 'median', 2: 'std_dev', 3: 'ac(1)'}, inplace=True)
    fig_initiate()
    fig_title("<<< Liquidity Measures in LOW vol regime -- Summary Statistics (Standardized Data) >>>")
    print(table2)
    fig_terminate()


def fig_title(astring):
    print(astring)
    print()


def fig_initiate():
    print()
    print("*******************************************************")


def fig_terminate():
    print("*******************************************************")
    print()
    print()


def reg_base(path, proxy, file):
    par = (Path(path)).parent
    mm = par / 'base_for_regression'
    os.chdir(str(mm))
    cols = ['EWPQS', 'EWDEPTH', 'return', 'V', 'SWPQS', 'SWPES', 'SWPTS', 'TWPQS',
            'MKT_EWPQS', 'MKT_EWDEPTH',
            'MKT_SWPQS', 'MKT_SWPES', 'MKT_SWPTS', 'MKT_TWPQS', 'MKT_return',
            'MKT_EWPQS_LAG1', 'MKT_EWPQS_LEAD1', 'MKT_EWDEPTH_LAG1',
            'MKT_EWDEPTH_LEAD1', 'MKT_SWPQS_LAG1', 'MKT_SWPQS_LEAD1', 'MKT_SWPES_LAG1',
            'MKT_SWPES_LEAD1', 'MKT_SWPTS_LAG1', 'MKT_SWPTS_LEAD1', 'MKT_TWPQS_LAG1',
            'MKT_TWPQS_LEAD1', 'MKT_return_LAG1', 'MKT_return_LEAD1']
    df = pd.read_csv(file, usecols=cols, encoding='latin_1')
    fig_title(f"{file} -- {proxy}")
    subcols = [c for c in cols if f"_{proxy}" in c]
    extracols = ['V', 'MKT_return', 'MKT_return_LAG1', 'MKT_return_LEAD1']
    [subcols.append(c) for c in extracols]
    results = sm.OLS(df[proxy] * 1000, sm.add_constant(df[subcols] * 1000)).fit()
    results = results.get_robustcov_results(cov_type='HAC', maxlags=1)

    # print(results.)
    print("           <<<< S U M M A R Y >>>")
    print(results.summary())
    print()
    explanatory = results.model.exog_names
    return results, explanatory


def tab4(results, explanatory):
    fig_initiate()
    fig_title("*** RESULTS <<< BETAS, P-VALUES, ADJ. R-SQUARED and WALD-TESTS>>> ***")
    # betas
    print(pd.DataFrame(data=results.params, index=explanatory, columns=['Coefficient']))
    print()
    # p-values
    print(pd.DataFrame(data=results.pvalues, index=explanatory, columns=['p-value']))
    print()
    # R²
    print(f"adj. R-squared: {results.rsquared_adj}")
    # Wald Test B1 +  B2 + B3 = 0
    hyp1 = f"({explanatory[1]} + {explanatory[2]} + {explanatory[3]} = 0)"
    wald_1 = results.wald_test(hyp1)
    print("Results for the 1st Wald Test Tab3")
    print(wald_1.summary())
    print()
    # Wald Test G = D1 = D2 = D3 = 0
    hyp2 = f"({explanatory[4]} = {explanatory[5]} = {explanatory[6]} = {explanatory[7]} = 0)"
    wald_2 = results.wald_test(hyp2)
    print("Results for the 2nd Wald Test Tab4")
    print(wald_2.summary())
    print()
    # F-Test
    print(f"F-test P-value : {results.f_pvalue}")
    fig_terminate()


def reg_conditionned(path, proxy, file):  # here endo and exo are stdzed (global-mu-and-sigma wise)
    par = (Path(path)).parent
    mm = par / 'base_for_regression'
    os.chdir(str(mm))
    cols = ['volatility', 'EWPQS', 'EWDEPTH', 'return', 'V', 'SWPQS', 'SWPES', 'SWPTS', 'TWPQS',
            'MKT_EWPQS', 'MKT_EWDEPTH', 'MKT_SWPQS', 'MKT_SWPES', 'MKT_SWPTS',
            'MKT_TWPQS', 'MKT_return', 'MKT_EWPQS_LAG1', 'MKT_EWPQS_LEAD1', 'MKT_EWDEPTH_LAG1',
            'MKT_EWDEPTH_LEAD1', 'MKT_SWPQS_LAG1', 'MKT_SWPQS_LEAD1', 'MKT_SWPES_LAG1',
            'MKT_SWPES_LEAD1', 'MKT_SWPTS_LAG1', 'MKT_SWPTS_LEAD1', 'MKT_TWPQS_LAG1',
            'MKT_TWPQS_LEAD1', 'MKT_return_LAG1', 'MKT_return_LEAD1']
    df = pd.read_csv(file, usecols=cols, encoding='latin_1')
    fig_title(f"{file} -- {proxy}")
    subcols = [f"volatility*{c}" for c in cols if f"_{proxy}" in c]
    extracols = ['volatility*V', 'volatility*MKT_return', 'volatility*MKT_return_LAG1', 'volatility*MKT_return_LEAD1']
    [subcols.append(c) for c in extracols]

    # results = sm.OLS(df[proxy] * 1000, sm.add_constant(df[subcols] * 1000)).fit()
    reststr = " + ".join(subcols)
    formx = f"{proxy} ~ {reststr}"
    print(formx)
    df[cols[1:]] = df[cols[1:]] * 1000
    df[cols[1:]] = (df[cols[1:]] - df[cols[1:]].mean()) / df[cols[1:]].std()
    results = smf.ols(formula=formx, data=df).fit()
    # Automatic lag truncation -->  Schwert (1989)
    lags = int(12 * (df.shape[0] / 100) ** (1 / 4))
    results = results.get_robustcov_results(cov_type='HAC', maxlags=lags)
    print("           <<<< S U M M A R Y >>>")
    print(results.summary())
    print()
    explanatory = results.model.exog_names
    return results, explanatory


def tab5(results, explanatory):
    fig_initiate()
    fig_title("*** RESULTS <<< BETAS, P-VALUES, ADJ. R-SQUARED and WALD-TESTS>>> ***")
    # betas
    print(pd.DataFrame(data=results.params, index=explanatory, columns=['Coefficient']))
    print()
    # Std. errors
    print(pd.DataFrame(data=results.bse, index=explanatory, columns=['Std. Errors']))
    print()
    # p-values
    print(pd.DataFrame(data=results.pvalues, index=explanatory, columns=['p-value']))
    print()
    # R²
    print(f"adj. R-squared: {results.rsquared_adj}")
    print()
    #  Test HO: B_high_vol -  B_low_vol > 0
    #       H1 ... B_high < B_low
    print()
    print(f"     Test    HO: Beta_{explanatory[3]} -  Beta_{explanatory[4]} > 0")
    print(f"             H1 ... Beta_{explanatory[3]} < Beta_{explanatory[4]}")
    print("where:   [T.L] --> low volatility variable")

    zhat = (results.params[3] - results.params[4]) / results.bse[3]
    pvalue = norm.cdf(np.abs(zhat))
    if pvalue < 0.05:
        print(f"P-value : {pvalue}")
        print("Reject Null Hypothesis in favor of H1")
    else:
        print(f"P-value : {pvalue}")
        print("Can NOT Reject Null Hypothesis")
    print()

    #  Test HO: B_high_vol -  B_low_vol < 0
    #       H1 ... B_high > B_low
    print(f"     Test    HO: Beta_{explanatory[3]} -  Beta_{explanatory[4]} < 0")
    print(f"             H1 ... Beta_{explanatory[3]} > Beta_{explanatory[4]}")
    print("where:   [T.L] --> low volatility variable")
    zhat = (results.params[4] - results.params[3]) / results.bse[4]
    pvalue = 1 - norm.cdf(np.abs(zhat))
    if pvalue < 0.05:
        print(f"P-value : {pvalue}")
        print("Reject Null Hypothesis in favor of H1")
    else:
        print(f"P-value : {pvalue}")
        print("Can NOT Reject Null Hypothesis")
    print()

    # F-Test
    print(f"F-test P-value : {results.f_pvalue}")
    fig_terminate()
