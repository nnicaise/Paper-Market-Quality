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
path_trades = "/home/hub/Downloads/Memoire-NICAISE-ANCIAUX/trades"
path_ob = "/home/hub/Downloads/Memoire-NICAISE-ANCIAUX/ob"

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
tmp = SAScsv(path_trades)
tmp.cp_to_child_folder()
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
