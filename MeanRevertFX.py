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
import parse_merge_FX_series
from sklearn.preprocessing import normalize
from sklearn.preprocessing import StandardScaler
import seaborn as sns
import DXY
import plotting_int_diff_fx_pairs
from sklearn.linear_model import LinearRegression
import statsmodels.tsa.stattools as ts
import numpy as np
import statsmodels.api
from scipy import stats
from sklearn import mixture as mix
from sklearn.svm import SVC

# import my parsing classes
prg = parse_merge_FX_series.parse_read_fx_series()
dxy = DXY.DXY()
P = plotting_int_diff_fx_pairs.Plot_rates_fx()


###############################################################################
# intelligent stat arb trad system
###############################################################################

class mean_revert_algo(object):

	def __init__(self, eurusd, gbpusd, carry_one, carry_two, carry_second_cc):
		self.eurusd = eurusd
		self.gbpusd = gbpusd
		self.carry_one = carry_one
		self.carry_two = carry_two
		self.carry_second_cc = carry_second_cc


	#### normalizing function
	def normal(self,eruusd_dxy):
		eur_dxy_ret= pd.DataFrame()
		eur_dxy_ret['Date'] = eruusd_dxy['Date']
		eur_dxy_ret['Price_x']=(eruusd_dxy['Price_x'].shift(1) - eruusd_dxy['Price_x'])/eruusd_dxy['Price_x']
		eur_dxy_ret['Price_y']=(eruusd_dxy['Price_y'].shift(1) - eruusd_dxy['Price_y'])/eruusd_dxy['Price_y']

		eur_dxy_norm= pd.DataFrame()
		eur_dxy_norm['Date'] = eur_dxy_ret['Date']
		eur_dxy_norm['Price_x'] = (1+eur_dxy_ret['Price_x']).cumprod()
		eur_dxy_norm['Price_y'] = (1+eur_dxy_ret['Price_y']).cumprod()
		eur_dxy_norm['Price_x'][:1] =1
		eur_dxy_norm['Price_y'][:1] =1
		eur_dxy_norm['Price_x'] = eur_dxy_norm['Price_x']*100
		eur_dxy_norm['Price_y'] = eur_dxy_norm['Price_y']*100
		return eur_dxy_norm

	def int_stat_arb(self,eurdxy_norm,n):
		eurdxy_norm = eurdxy_norm.dropna()
		eurdxy_norm['Ratio'] = eurdxy_norm['Price_x']/eurdxy_norm['Price_y']
		eurdxy_norm['Beta'] = eurdxy_norm['Ratio'].rolling(window=n).mean()
		eurdxy_norm['Alpha'] = eurdxy_norm['Price_x'].rolling(window=n).mean() - eurdxy_norm['Beta']*eurdxy_norm['Price_y'].rolling(window=50).mean()
		eurdxy_norm['Z'] = eurdxy_norm['Price_x'] - eurdxy_norm['Alpha'] -eurdxy_norm['Beta']*eurdxy_norm['Price_y']
		resid_t = eurdxy_norm['Price_x'] - eurdxy_norm['Alpha'] -eurdxy_norm['Beta']*eurdxy_norm['Price_y']
		return eurdxy_norm


	def plot_res_mat(self,eurdxy_norm):
		eurdxy_norm_per = eurdxy_norm[eurdxy_norm['Date'] > datetime.datetime(2009,1,1)]
		fig, ax1 = plt.subplots()
		x=eurdxy_norm_per['Date']
		y1=eurdxy_norm_per['Price_x']
		y2=eurdxy_norm_per['Price_y']
		y3=eurdxy_norm_per['Z']
		ax2 = ax1.twinx()
		ax1.plot(x, y1, 'g-')
		ax1.plot(x, y2, 'b-')
		ax2.plot(x, y3, 'r-')
		return plt.show()

	#################################################################################
	#eurusd gbpusd
	def merging_func(self):
		eurusd = self.eurusd.dropna()
		gbpusd = self.gbpusd.dropna()
		eurusd_gbpusd = P.match_matrix_FX(eurusd,gbpusd,'Price')
		return eurusd_gbpusd

	def lag_function(self,eurusd_gbpusd,n):
		eurusd_gbpusd_norm = self.normal(eurusd_gbpusd)
		eurusd_gbpusd_norm = self.int_stat_arb(eurusd_gbpusd_norm,n)
		self.plot_res_mat(eurusd_gbpusd_norm)
		ts.adfuller(eurusd_gbpusd_norm['Z'].dropna())
		eurusd_gbpusd_norm.plot(x='Date',y='Z')
		return eurusd_gbpusd_norm

	########################### mixture #############################
	def mixture_gauging(self,eurusd_gbpusd_norm):
		test = eurusd_gbpusd_norm[['Date','Z']]
		n= 10
		t= 1
		split = int(t*len(test))

		ss = StandardScaler()
		unsup = mix.GaussianMixture(n_components=10,
		                            covariance_type="spherical",
		                            n_init=100,
		                            random_state=42)

		test= test.dropna()
		test = test.set_index('Date')
		unsup.fit(np.reshape(ss.fit_transform(test[:split]),(-1,test.shape[1])))
		regime = unsup.predict(np.reshape(ss.fit_transform(test[:split]),(-1,test.shape[1])))
		Regimes = pd.DataFrame(regime,columns=['Regime'],index=test[:split].index).join(test[:split], how='inner').reset_index(drop=False).rename(columns={'index':'Date'})

		order=[0,1,2,3,4,5,6,7,8,9]
		fig = sns.FacetGrid(data=Regimes,hue='Regime',hue_order=order,aspect=2,size=4)
		fig.map(plt.scatter,'Date','Z',s=4).add_legend()
		plt.show()

		fd=[g for g in unsup.means_]
		for i in order:
			print('Mean for regime %i: '%i,unsup.means_[i][0])
		mat = pd.DataFrame({'Regime':order,'Mean_Regime':fd})
		mat = mat.sort_values('Mean_Regime')

		return mat,Regimes
	################# end of gaussian mixture part ###################
	##################################################################

	def mat_cal(self,eurusd_gbpusd,eurusd_gbpusd_norm,Regimes):
		mat = pd.concat([eurusd_gbpusd,eurusd_gbpusd_norm[['Beta','Z']],Regimes['Regime']],axis=1)
		mat['Regime_lag'] = mat['Regime'].shift(1)
		mat['Beta_lag'] = mat['Beta'].shift(1)
		mat['Z_lag'] = mat['Z'].shift(1)
		mat = mat.dropna()
		return mat

	############################ Short Leg ############################
	def match_gain(self,df1,df2):
		try:
			df1['Num_Date'] = pd.to_numeric(df1['Date'])
			df2['Num_Date'] = pd.to_numeric(df2['Date'])
			t = pd.DataFrame()
			for i in range(0,len(df1['Date'])):
				s = [g - df1['Num_Date'].iloc[i] for g in df2['Num_Date']]
				df2['Diff'] = s
				ddf2 = df2[df2['Diff']>0]
				smin = min(ddf2['Diff'])
				dg = df2[df2['Diff']==smin]
				t = pd.concat([t,dg])
		except Exception as e:
			print('type error: ' +str(e))
		return t


	def short_leg(self,mat,matr):
		# EURUSD - GBPUSD
		overnight_rates = self.overnight_carry_mat()
		carry_rates = self.select_carry_rates(overnight_rates)

		moment1 = mat.loc[(mat['Regime']==matr['Regime'].iloc[1]) & (mat['Regime_lag']==matr['Regime'].iloc[2])]
		moment2 = mat.loc[(mat['Regime']==matr['Regime'].iloc[7]) & (mat['Regime_lag']==matr['Regime'].iloc[6])]
		moment1['Date_month'] = pd.to_datetime(moment1['Date']).map(lambda x: x.strftime('%Y-%m'))
		moment2['Date_month'] = pd.to_datetime(moment2['Date']).map(lambda x: x.strftime('%Y-%m'))

		mom_match = self.match_gain(moment1,moment2)

		# merging together thetwo dataframes
		moment1['index'] = range(0,len(moment1['Date']))
		mom_match['index'] = range(0,len(mom_match['Date']))
		moment1.set_index('index', inplace=True)
		mom_match.set_index('index', inplace=True)
		merg = pd.merge(moment1,mom_match, how='inner', left_index=True, right_index=True).reset_index()

		# overnight_rates = self.overnight_carry_mat()
		# carry_rates = self.select_carry_rates(overnight_rates)

		clean_merg = merg[['Date_x','Date_month_x','Price_x_x','Price_y_x','Beta_x','Date_y','Date_month_y','Price_x_y','Price_y_y','Beta_y']]
		clean_merg['Beta_x*Price_y_x'] = clean_merg['Beta_x']*clean_merg['Price_y_x']
		clean_merg['Beta_x*Price_y_y'] = clean_merg['Beta_x']*clean_merg['Price_y_y']

	    # strategy short Beta*y and long x
		clean_merg['y_leg'] = clean_merg['Beta_x*Price_y_x'] - clean_merg['Beta_x*Price_y_y']
		clean_merg['y_leg_return'] = (clean_merg['Price_y_x'] - clean_merg['Price_y_y'])/clean_merg['Price_y_y']
		clean_merg['Beta*y_leg_return'] = clean_merg['Beta_x']*clean_merg['y_leg_return']
		clean_merg['x_leg'] = clean_merg['Price_x_y'] - clean_merg['Price_x_x']
		clean_merg['x_leg_return'] = (clean_merg['Price_x_y'] - clean_merg['Price_x_x'])/clean_merg['Price_x_x']
		clean_merg['Result'] = clean_merg['y_leg'] + clean_merg['x_leg']
		clean_merg['Result_return'] = clean_merg['Beta*y_leg_return'] + clean_merg['x_leg_return']
		clean_merg['Carry'] = self.calculating_carry_shortbetay_longx(clean_merg,carry_rates)
		clean_merg['Carry'] = clean_merg.Carry/100
		clean_merg['Result_return_final'] = clean_merg.Result_return + clean_merg.Carry


		short_leg = clean_merg[['Date_x','Result','Result_return','Carry','Result_return_final']]
		short_leg['Cumsum_Result'] = np.cumsum(short_leg['Result_return'])
		short_leg['Cumsum_Result_return_final'] = np.cumsum(short_leg['Result_return_final'])
		short_leg.plot(x='Date_x',y='Cumsum_Result_return_final', title='Cumulative Gains short Leg')

		return clean_merg,short_leg
	############################ Long Leg ############################

	def long_leg(self,mat,matr):

		overnight_rates = self.overnight_carry_mat()
		carry_rates = self.select_carry_rates(overnight_rates)

		moment3 = mat.loc[(mat['Regime']==matr['Regime'].iloc[8]) & (mat['Regime_lag']==matr['Regime'].iloc[7])]
		moment4 = mat.loc[(mat['Regime']==matr['Regime'].iloc[2]) & (mat['Regime_lag']==matr['Regime'].iloc[3])]
		moment3['Date_month'] = pd.to_datetime(moment3['Date']).map(lambda x: x.strftime('%Y-%m'))
		moment4['Date_month'] = pd.to_datetime(moment4['Date']).map(lambda x: x.strftime('%Y-%m'))

		mom_match_lg = self.match_gain(moment3,moment4)

		# merging together thetwo dataframes
		moment3['index'] = range(0,len(moment3['Date']))
		mom_match_lg['index'] = range(0,len(mom_match_lg['Date']))
		moment3.set_index('index', inplace=True)
		mom_match_lg.set_index('index', inplace=True)
		merg_lg = pd.merge(moment3,mom_match_lg, how='inner', left_index=True, right_index=True).reset_index()

		clean_merg_lg = merg_lg[['Date_x','Date_month_x','Price_x_x','Price_y_x','Beta_x','Date_y','Date_month_y','Price_x_y','Price_y_y','Beta_y']]
		clean_merg_lg['Beta_x*Price_y_x'] = clean_merg_lg['Beta_x']*clean_merg_lg['Price_y_x']
		clean_merg_lg['Beta_x*Price_y_y'] = clean_merg_lg['Beta_x']*clean_merg_lg['Price_y_y']

	    # strategy long Beta*y and Short x
		clean_merg_lg['y_leg'] =  clean_merg_lg['Beta_x*Price_y_y'] - clean_merg_lg['Beta_x*Price_y_x']
		clean_merg_lg['y_leg_return'] = (clean_merg_lg['Price_y_y'] - clean_merg_lg['Price_y_x'])/clean_merg_lg['Price_y_x']
		clean_merg_lg['Beta*y_leg_return'] = clean_merg_lg['Beta_x'] * clean_merg_lg['y_leg_return']
		clean_merg_lg['x_leg'] =  clean_merg_lg['Price_x_x'] - clean_merg_lg['Price_x_y']
		clean_merg_lg['x_leg_return'] =  (clean_merg_lg['Price_x_x'] - clean_merg_lg['Price_x_y'])/clean_merg_lg['Price_x_y']
		clean_merg_lg['Result'] = clean_merg_lg['y_leg'] + clean_merg_lg['x_leg']
		clean_merg_lg['Result_return'] = clean_merg_lg['Beta*y_leg_return'] + clean_merg_lg['x_leg_return']
		clean_merg_lg['Carry'] = self.calculating_carry_longbetay_shortx(clean_merg_lg,carry_rates)
		clean_merg_lg['Carry'] = clean_merg_lg.Carry/100
		clean_merg_lg['Result_return_final'] = clean_merg_lg.Result_return + clean_merg_lg.Carry


		long_leg = clean_merg_lg[['Date_x','Result','Result_return','Result_return_final']]
		long_leg['Cumsum_Result'] = np.cumsum(long_leg['Result'])
		long_leg['Cumsum_Result_return_final'] = np.cumsum(long_leg['Result_return_final'])
		long_leg.plot(x='Date_x',y='Cumsum_Result_return_final', title='Cumulative Gains long Leg')

		return(clean_merg_lg,long_leg)


	def overnight_carry_mat(self):
		os.chdir(r'C:\Users\Federico\Documents\Trading\Algo')
		rates = pd.read_csv('CB_rates.csv',error_bad_lines=False,sep=';')
		rates.Date = pd.to_datetime(rates.Date)
		rates.columns = ['Date','UK','EUR','US','JPY','AUS','NZD','SWI','CAD','SWE','BRA','INR']
		rates['Date_month'] = pd.to_datetime(rates['Date']).map(lambda x: x.strftime('%Y-%m'))
		sd = {'Date':pd.date_range(start='2000-01-01', end='2018-09-01',freq='M').map(lambda x: x.strftime('%Y-%m'))}
		sd = pd.DataFrame(sd)
		sd.set_index('Date',inplace=True)
		rates.set_index('Date_month',inplace=True)
		sd = pd.merge(sd,rates[['UK','EUR','US','JPY','AUS','NZD','SWI','CAD','SWE','BRA','INR']],how='left', left_index=True, right_index=True)
		sd = sd.fillna(method='ffill')
		def replace_comma(x):
			try:
				return x.replace(',','.')
			except AttributeError:
				return np.NaN
		sd['UK'] = [replace_comma(strit) for strit in sd['UK']]
		sd['EUR'] = [replace_comma(strit) for strit in sd['EUR']]
		sd['US'] = [replace_comma(strit) for strit in sd['US']]
		sd['JPY'] = [replace_comma(strit) for strit in sd['JPY']]
		sd['AUS'] = [replace_comma(strit) for strit in sd['AUS']]
		sd['NZD'] = [replace_comma(strit) for strit in sd['NZD']]
		sd['SWI'] = [replace_comma(strit) for strit in sd['SWI']]
		sd['CAD'] = [replace_comma(strit) for strit in sd['CAD']]
		sd['SWE'] = [replace_comma(strit) for strit in sd['SWE']]
		sd['BRA'] = [replace_comma(strit) for strit in sd['BRA']]
		sd['INR'] = [replace_comma(strit) for strit in sd['INR']]
		sd['UK_daily'] = pd.to_numeric(sd['UK'])/365
		sd['EUR_daily'] = pd.to_numeric(sd['EUR'])/365
		sd['US_daily'] = pd.to_numeric(sd['US'])/365
		sd['JPY_daily'] = pd.to_numeric(sd['JPY'])/365
		sd['AUS_daily'] = pd.to_numeric(sd['AUS'])/365
		sd['NZD_daily'] = pd.to_numeric(sd['NZD'])/365
		sd['SWI_daily'] = pd.to_numeric(sd['SWI'])/365
		sd['CAD_daily'] = pd.to_numeric(sd['CAD'])/365
		sd['SWE_daily'] = pd.to_numeric(sd['SWE'])/365
		sd['BRA_daily'] = pd.to_numeric(sd['BRA'])/365
		sd['INR_daily'] = pd.to_numeric(sd['INR'])/365
		sd = sd.reset_index()
		return sd

	def select_carry_rates(self,rates):
		df_n = rates[['Date',self.carry_one,self.carry_two,self.carry_second_cc,self.carry_one+'_daily',
						self.carry_two+'_daily',self.carry_second_cc+'_daily']]
		return df_n

	def calculating_carry_shortbetay_longx(self,clean_merg,carry_rates):
		clean_merg = clean_merg.reset_index()
		dates = clean_merg[['Date_x','Date_y','Date_month_x','Date_month_y','Beta_x']]
		dates['Dates_diff'] = dates.Date_y - dates.Date_x
		dates['Dates_diff_num'] = [float(trit.days) for trit in dates.Dates_diff]
		carry = []
		carry_rates.Date = pd.to_datetime(carry_rates.Date).map(lambda x: x.strftime('%Y-%m'))
		dates.Date_x = pd.to_datetime(dates.Date_x).map(lambda x: x.strftime('%Y-%m'))
		dates.Date_y = pd.to_datetime(dates.Date_y).map(lambda x: x.strftime('%Y-%m'))
		for i in range(0,len(dates['Date_x'])):
		    # strategy short Beta*y and long x
			temp = carry_rates[(carry_rates.Date>=dates['Date_x'].iloc[i]) & (carry_rates.Date <= dates['Date_y'].iloc[i])]
			temp['Beta'] = dates['Beta_x'].iloc[i]
			temp[self.carry_second_cc+'_minus_'+self.carry_two] = temp[self.carry_second_cc+'_daily'] - temp[self.carry_two+'_daily']
			temp['Beta_'+self.carry_second_cc+'_minus_'+self.carry_two] = temp[self.carry_second_cc+'_minus_'+self.carry_two] * temp['Beta']
			temp[self.carry_one+'_minus_'+self.carry_second_cc] = temp[self.carry_one+'_daily'] - temp[self.carry_second_cc+'_daily']
			temp['carry_daily'] = temp['Beta_'+self.carry_second_cc+'_minus_'+self.carry_two] + temp[self.carry_one+'_minus_'+self.carry_second_cc]
			temp['carry_monthly'] = temp['carry_daily'] * 30
			carry.append(sum(temp['carry_monthly']))
		return carry

	def calculating_carry_longbetay_shortx(self,clean_merg,carry_rates):
		dates = clean_merg[['Date_x','Date_y','Date_month_x','Date_month_y','Beta_x']]
		dates['Dates_diff'] = dates.Date_y - dates.Date_x
		dates['Dates_diff_num'] = [float(trit.days) for trit in dates.Dates_diff]
		carry = []
		carry_rates.Date = pd.to_datetime(carry_rates.Date).map(lambda x: x.strftime('%Y-%m'))
		dates.Date_x = pd.to_datetime(dates.Date_x).map(lambda x: x.strftime('%Y-%m'))
		dates.Date_y = pd.to_datetime(dates.Date_y).map(lambda x: x.strftime('%Y-%m'))
		for i in range(0,len(dates['Date_x'])):
		    # strategy long Beta*y and Short x
			temp = carry_rates[(carry_rates.Date>=dates['Date_x'].iloc[i]) & (carry_rates.Date <= dates['Date_y'].iloc[i])]
			temp['Beta'] = dates['Beta_x'].iloc[i]
			temp[self.carry_two+'_minus_'+self.carry_second_cc] = temp[self.carry_two+'_daily'] - temp[self.carry_second_cc+'_daily']
			temp['Beta_'+self.carry_two+'_minus_'+self.carry_second_cc] = temp[self.carry_two+'_minus_'+self.carry_second_cc] * temp['Beta']
			temp[self.carry_second_cc+'_minus_'+self.carry_one] = temp[self.carry_second_cc+'_daily'] - temp[self.carry_one+'_daily']
			temp['carry_daily'] = temp['Beta_'+self.carry_two+'_minus_'+self.carry_second_cc] + temp[self.carry_second_cc+'_minus_'+self.carry_one]
			temp['carry_monthly'] = temp['carry_daily'] * 30
			carry.append(sum(temp['carry_monthly']))
		return carry

