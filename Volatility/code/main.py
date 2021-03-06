from modules.convertSAScsv import SAScsv
from modules.preprocess import preprocess_trades_ob as pp
from modules.wavg import weihted_averages as wavg
from modules.market_variables import market_variables as mv
import modules.regressions_tables as models
"""
============================================================================
PARAMETERS: user's manual INPUT
============================================================================
"""

"""
path to trades and ob files
"""
path_trades = "C:/Users/NicolasNicaise/OneDrive - Agilytic/Documents/Papers/Volatility/trades"
path_ob = "C:/Users/NicolasNicaise/OneDrive - Agilytic/Documents/Papers/Volatility/ob"

filenames_list = ["bitfinex_bchusd.csv",
                  "bitfinex_btcusd.csv",
                  "bitfinex_eosusd.csv",
                  "bitfinex_ethusd.csv",
                  "bitfinex_xrpusd.csv"]

pl = ['EWPQS', 'EWDEPTH',
      'SWPQS', 'SWPES',
      'SWPTS', 'TWPQS']

"""
============================================================================
PART I  : CONVERT, CLEAN, REDUCE, COMPUTE data
============================================================================
"""

"""
[PROD] convert SAS--> CSV + move to subfolder
"""
#tmp = SAScsv(path_trades)
#tmp.cp_to_child_folder()
tmp = SAScsv(path_ob)
tmp.cp_to_child_folder()

"""
[PROD] clean data
"""
[pp(path_trades, filename) for filename in filenames_list]

"""
[PROD] weighted averages proxies + VOL flag
"""
[wavg(path_trades, filename) for filename in filenames_list]

"""
[PROD] market-wide variables
"""
mv(path_trades, filenames_list)


"""
============================================================================
PART II  : RESULTS
============================================================================
"""

"""
[PROD] Table 2 --> Aggregate Liquidity Measure -- Summary Statistics
"""
#models.agg_liq_measures(path_trades)


"""
[PROD] Table 3 Aggregate Stfzed Liquidity Measure -- per VOL
"""
#models.stdzed_liq_measures_per_vol_regime(path_trades)


"""
 [PROD] -- > table 4 using Newey & West (1994) HAC robust std errors
 INPUTS: cryptocurrency & proxy
       example
               file = 'bitfinex_xrpusd.csv'
                proxy = 'EWPQS'

"""
#file = 'bitfinex_xrpusd.csv'
#proxy = 'EWPQS'
#results, explanatory = models.reg_base(path_trades, proxy, file)
#models.tab4(results, explanatory)


"""
 [PROD] --> table 5 conditionned on vol regime using Newey & West (1994) HAC robust std errors
 INPUTS: cryptocurrency & proxy
       example
               file = 'bitfinex_xrpusd.csv'
                proxy = 'EWPQS'
"""
#file = 'bitfinex_xrpusd.csv'
#proxy = 'EWPQS'
#results, explanatory = models.reg_conditionned(path_trades, proxy, file)
#models.tab5(results, explanatory)


''' TESTS DE STATIONARITE 

from pathlib import Path
import pandas as pd
from statsmodels.tsa.stattools import adfuller


par = (Path(path)).parent
mm = par / 'base_for_regression'
stationarity = par / 'stationarity'
os.chdir(str(mm))
os.makedirs(str(stationarity))
final = []

for f in os.listdir() :
    for proxy in ['EWPQS', 'EWDEPTH', 'SWPQS', 'SWPTS', 'SWPES', 'TWPQS'] :
        df = pd.read_csv(f)
        adf_test = adfuller(df[proxy])
        
        crypto = f.replace("bitfinex_", "").replace(".csv", "")
        results = [crypto, proxy, adf_test[0], adf_test[1], adf_test[2], adf_test[3], adf_test[4]["1%"], 
        adf_test[4]["5%"], adf_test[4]["10%"]]
        final.append(results)

      
        
mycolumns = ["Crypto", "Proxy", "Test_statistic", "MacKinnons_approximate_p-value", "Number_of_lags_used",
"Number of obs", "Critical values 1%", "Critical values 5%", "Critical values 10%"]
output = pd.DataFrame(final, columns=mycolumns)
str_output = str(stationarity) + "\\stationarity_tests.xlsx"
output.to_excel (str_output, index= False)

'''