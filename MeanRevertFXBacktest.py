#running some ideas
import os
#os.chdir(r'C:\Users\Federico\Documents\Trading')

import urllib.request
import pickle
from pynput import mouse
from pynput import keyboard
from ctypes import windll
from time import sleep
import pandas as pd
import pyautogui
import os
from bs4 import BeautifulSoup
import unicodedata
import datetime
import re
import matplotlib.pyplot as plt
import requests
from textblob import TextBlob
from itertools import repeat
import ParseMergeFXSeries
from sklearn.preprocessing import normalize
from sklearn.preprocessing import StandardScaler
import seaborn as sns
import DXY
import PlottingIntDiffFXPairs
from sklearn.linear_model import LinearRegression
import statsmodels.tsa.stattools as ts
import numpy as np
import statsmodels.api
from scipy import stats
from sklearn import mixture as mix
from sklearn.svm import SVC
import MeanRevertFX
import collections
import statsmodels.api as sm
from johansen import coint_johansen
from datetime import datetime, timedelta
from collections import OrderedDict

# import my parsing
prg = parse_merge_FX_series.parse_read_fx_series()
dxy = DXY.DXY()
P = plotting_int_diff_fx_pairs.Plot_rates_fx()

#### Try on EURUSD and usdcad (daily backtest)
#EURUSD
Eurusd_old = prg.parse('EURUSD')
Eurusd_old = Eurusd_old[['Date','High','Low','Open','Price']]
Eurusd_new = prg.scrap('eur','usd')
eurusd = prg.merge(Eurusd_old,Eurusd_new,'EURUSD.csv')

#GBPUSD
gbpusd_old = prg.parse('GBPUSD')
gbpusd_old = gbpusd_old[['Date','High','Low','Open','Price']]
gbpusd_new = prg.scrap('gbp','usd')
gbpusd = prg.merge(gbpusd_old,gbpusd_new,'GBPUSD.csv')


### trying ADF and CADF and Johansen Test (do not work) ###
MR_one = mean_revert_FX_one.mean_revert_algo(eurusd,gbpusd,'EUR','UK','US')
eurusd_gbpusd = MR_one.merging_func()

fig, ax1 = plt.subplots()
x=eurusd_gbpusd['Date']
y1=eurusd_gbpusd['Price_x']
y2=eurusd_gbpusd['Price_y']
ax2 = ax1.twinx()
ax1.plot(x, y1, 'g-')
ax1.plot(x, y2, 'b-')

eurusd_gbpusd.Price_x = pd.to_numeric(eurusd_gbpusd.Price_x)
eurusd_gbpusd.Price_y = pd.to_numeric(eurusd_gbpusd.Price_y)

ts.adfuller(eurusd_gbpusd['Price_y'])
result = ts.coint(eurusd_gbpusd['Price_x'],eurusd_gbpusd['Price_y'])

eurusd_gbpusd_test = eurusd_gbpusd[['Price_x','Price_y']]

coint_johansen(eurusd_gbpusd_test, -1, 1)

### Analyzing spreads and ratios ###

eurusd_gbpusd['Price_diff'] = eurusd_gbpusd.Price_x - eurusd_gbpusd.Price_y
eurusd_gbpusd['Log_ret'] = np.log(eurusd_gbpusd.Price_x/eurusd_gbpusd.Price_y)
eurusd_gbpusd['ret'] = eurusd_gbpusd.Price_x/eurusd_gbpusd.Price_y

eurusd_gbpusd.plot('Date','ret')
plt.show()

########################################

eurusd_gbpusd_norm = MR_one.lag_function(eurusd_gbpusd,60)
matr,Regimes = MR_one.mixture_gauging(eurusd_gbpusd_norm)
df = MR_one.mat_cal(eurusd_gbpusd,eurusd_gbpusd_norm,Regimes)
short_summary,short_leg_mat = MR_one.short_leg(df,matr)
long_summary,long_leg_mat = MR_one.long_leg(df,matr)


######### AUDUSD and CADUSD #############

#AUDUSD
audusd_old = prg.parse('AUDUSD')
audusd_old = audusd_old[['Date','High','Low','Open','Price']]
audusd_new = prg.scrap('aud','usd')
audusd = prg.merge(audusd_old,audusd_new,'AUDUSD.csv')

#USDCAD
usdcad_old = prg.parse('USDCAD')
usdcad_old = usdcad_old[['Date','High','Low','Open','Price']]
usdcad_new = prg.scrap('usd','cad')
usdcad = prg.merge(usdcad_old,usdcad_new,'USDCAD.csv')

cadusd = usdcad.copy()
cadusd.Price = 1/cadusd.Price

MR_two = mean_revert_FX_one.mean_revert_algo(audusd,cadusd,'AUS','CAD','US')
audusd_cadusd = MR_two.merging_func()
audusd_usdcad_norm = MR_two.lag_function(audusd_cadusd,60)
matr_reg_audcad,Regimes_audcad = MR_two.mixture_gauging(audusd_usdcad_norm)
df_audcad = MR_two.mat_cal(audusd_cadusd,audusd_usdcad_norm,Regimes_audcad)
short_summary_audcad,short_leg_mat_audcad = MR_two.short_leg(df_audcad,matr_reg_audcad)
long_summary_audcad,long_leg_mat_audcad = MR_two.long_leg(df_audcad,matr_reg_audcad)


###### NZDUSD AUDUSD ########
#AUDUSD
audusd_old = prg.parse('AUDUSD')
audusd_old = audusd_old[['Date','High','Low','Open','Price']]
audusd_new = prg.scrap('aud','usd')
audusd = prg.merge(audusd_old,audusd_new,'AUDUSD.csv')

#NZDUSD
nzdusd_old = prg.parse('NZDUSD')
nzdusd_old = nzdusd_old[['Date','High','Low','Open','Price']]
nzdusd_new = prg.scrap('nzd','usd')
nzdusd = prg.merge(nzdusd_old,nzdusd_new,'NZDUSD.csv')


MR_three = mean_revert_FX_one.mean_revert_algo(audusd,nzdusd,'AUS','NZD','US')
audusd_nzdusd = MR_three.merging_func()
audusd_nzdusd_norm = MR_three.lag_function(audusd_nzdusd,60)
matr_reg_audnzd,Regimes_audnzd = MR_three.mixture_gauging(audusd_nzdusd_norm)
df_audnzd = MR_three.mat_cal(audusd_nzdusd,audusd_nzdusd_norm,Regimes_audnzd)
short_summary_audnzd,short_leg_mat_audnzd = MR_three.short_leg(df_audnzd,matr_reg_audnzd)
long_summary_audnzd,long_leg_mat_audnzd = MR_three.long_leg(df_audnzd,matr_reg_audnzd)


###### NZDUSD and EURUSD ########
eurusd = eurusd.dropna()
nzdusd = nzdusd.dropna()

MR_four = mean_revert_FX_one.mean_revert_algo(eurusd,nzdusd,'EUR','NZD','US')
eurusd_nzdusd = MR_four.merging_func()
eurusd_nzdusd_norm = MR_four.lag_function(eurusd_nzdusd,60)
matr_reg_eurnzd,Regimes_eurnzd = MR_four.mixture_gauging(eurusd_nzdusd_norm)
df_eurnzd = MR_four.mat_cal(eurusd_nzdusd,eurusd_nzdusd_norm,Regimes_eurnzd)
short_summary_eurnzd,short_leg_mat_eurnzd = MR_four.short_leg(df_eurnzd,matr_reg_eurnzd)
long_summary_eurnzd,long_leg_mat_eurnzd = MR_four.long_leg(df_eurnzd,matr_reg_eurnzd)

##### EURUSD and USDJPY #########
Usdjpy_old = prg.parse('USDJPY')
Usdjpy_old = Usdjpy_old[['Date','High','Low','Open','Price']]
Usdjpy_new = prg.scrap('usd','jpy')
usdjpy = prg.merge(Usdjpy_old,Usdjpy_new,'USDJPY.csv')

jpyusd = usdjpy.copy()
jpyusd.Price = 1/jpyusd.Price


MR_five = mean_revert_FX_one.mean_revert_algo(eurusd,jpyusd,'EUR','JPY','US')
eurusd_jpyusd = MR_five.merging_func()
eurusd_jpyusd_norm = MR_five.lag_function(eurusd_jpyusd,60)
matr_reg_eurjpy,Regimes_eurjpy = MR_five.mixture_gauging(eurusd_jpyusd_norm)
df_eurjpy = MR_five.mat_cal(eurusd_jpyusd,eurusd_jpyusd_norm,Regimes_eurjpy)
short_summary_eurjpy,short_leg_mat_eurjpy = MR_five.short_leg(df_eurjpy,matr_reg_eurjpy)
long_summary_eurjpy,long_leg_mat_eurjpy = MR_five.long_leg(df_eurjpy,matr_reg_eurjpy)


####### EURUSD and AUDUSD ########
MR_six = mean_revert_FX_one.mean_revert_algo(eurusd,audusd,'EUR','AUS','US')
eurusd_audusd = MR_six.merging_func()
eurusd_audusd_norm = MR_six.lag_function(eurusd_audusd,60)
matr_reg_euraud,Regimes_euraud = MR_six.mixture_gauging(eurusd_audusd_norm)
df_euraud = MR_six.mat_cal(eurusd_audusd,eurusd_audusd_norm,Regimes_euraud)
short_summary_euraud,short_leg_mat_euraud = MR_six.short_leg(df_euraud,matr_reg_euraud)
long_summary_euraud,long_leg_mat_euraud = MR_six.long_leg(df_euraud,matr_reg_euraud)
