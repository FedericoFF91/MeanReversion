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
import webbrowser
import zipfile
import shutil
import us_int_rates_class 
import japan_int_rates_class 
import germany_int_rates_class 
import FX_pairs as FX

class PlotRatesFx():
    
    def __init(self):
        pass

    @classmethod   
    def MatchDatasetInterestDifferential(self,MatrixOne,MatrixTwo,Ticker_one,ticker_two,japan=None):
        # select the tenor that you want
        SubsetDf = MatrixOne.filter(regex = '10Y')
        # adding the Date
        one = matrix_one[['Date',ticker_one]].dropna()
        two = matrix_two[matrix_two['Security']== ticker_two].dropna()
        one.set_index('Date', inplace=True)
        two.set_index('Date', inplace=True)
        merge = pd.merge(two,one, how='inner', left_index=True, right_index=True)
        merge = merge.reset_index()
        merge.rename(columns={'index':'Date'}, inplace=True)
        merge['Date'] = pd.to_datetime(merge['Date'])
        merge['Difference'] = merge['Value'] - merge['10Y']
        return merge
        # else:
        #     one = matrix_one[matrix_one['Security']==ticker_one].dropna()
        #     two = matrix_two[matrix_two['Security']==ticker_two].dropna()
        #     one.set_index('Date', inplace=True)
        #     two.set_index('Date', inplace=True)
        #     merge = pd.merge(two,one, how='inner', left_index=True, right_index=True)
        #     merge = merge.reset_index()
        #     merge.rename(columns={'index':'Date'}, inplace=True)
        #     merge['Date'] = pd.to_datetime(merge['Date'])
        #     merge['Difference'] = merge['Value_x'] - merge['Value_y']
        #     return merge
        
    def plotting_matrix_g(self,merge,ticker_one,japan=None):
        if japan == 'True':
            merge = merge.reset_index()
            t = merge.plot(x='Date',y=['Difference'],title='Spread US-Japan, Difference between '+ticker_one)
            return t
        else:
            t = merge.plot(x='Date',y=['Difference'],title='Spread US-Germany, Difference between '+ticker_one)
            return t
    
    def get_time_mat_g(self,date,merge):
        merge = merge[merge['Date'] > datetime.datetime.strptime(date, '%Y-%m-%d')]
        return merge.dropna()

    def matching_diff_fx(self, merge,fx_mat,ticker_match):
        m = fx_mat[['Results',ticker_match]]
        merge.set_index('Date', inplace=True)
        m.set_index('Results', inplace=True)
        mm = pd.merge(merge,m, how='inner', left_index=True, right_index=True).reset_index()
        mm.rename(columns={'index':'Date'}, inplace=True)
        mm['Date'] = pd.to_datetime(mm['Date'])
        return mm

    def matching_fx_interest_rates(self, merge,fx_mat,variable):
        m = fx_mat[['Date',variable]]
        merge.set_index('Date', inplace=True)
        m.set_index('Date', inplace=True)
        mm = pd.merge(merge,m, how='inner', left_index=True, right_index=True).reset_index()
        mm.rename(columns={'index':'Date'}, inplace=True)
        mm['Date'] = pd.to_datetime(mm['Date'])
        return mm

    def plot_mat(self,mat,ticker_match):
        fig, ax1 = plt.subplots()
        color = 'tab:red'
        ax1.set_xlabel('Date')
        ax1.set_ylabel('Difference between int rates', color=color)
        ax1.plot(mat['Date'], mat['Difference'], color=color)
        ax1.tick_params(axis='y', labelcolor=color)
        ax2 = ax1.twinx()  # instantiate a second axes that shares the same x-axis
        color = 'tab:blue'
        ax2.set_ylabel(ticker_match, color=color)  # we already handled the x-label with ax1
        ax2.plot(mat['Date'], mat[ticker_match], color=color)
        ax2.tick_params(axis='y', labelcolor=color)
        fig.tight_layout()  # otherwise the right y-label is slightly clipped
        plt.show()

    def match_matrix_FX(self,matrix_one,matrix_two,ticker_one):
        one = matrix_one[['Date',ticker_one]].dropna()
        two = matrix_two[['Date',ticker_one]].dropna()
        one.set_index('Date', inplace=True)
        two.set_index('Date', inplace=True)
        merge = pd.merge(one,two, how='inner', left_index=True, right_index=True)
        merge = merge.reset_index()
        merge.rename(columns={'index':'Date'}, inplace=True)
        merge['Date'] = pd.to_datetime(merge['Date'])
        return merge

    def plot_FX(self,mat,x_name,y_name):
        fig, ax1 = plt.subplots()
        color = 'tab:red'
        ax1.set_xlabel('Date')
        ax1.set_ylabel(x_name, color=color)
        ax1.plot(mat['Date'], mat['Price_x'], color=color)
        ax1.tick_params(axis='y', labelcolor=color)
        ax2 = ax1.twinx()  # instantiate a second axes that shares the same x-axis
        color = 'tab:blue'
        ax2.set_ylabel(y_name, color=color)  # we already handled the x-label with ax1
        ax2.plot(mat['Date'], mat['Price_y'], color=color)
        ax2.tick_params(axis='y', labelcolor=color)
        fig.tight_layout()  # otherwise the right y-label is slightly clipped
        plt.show()


    def match_matrix_FX_int_rates(self,matrix_one,matrix_two,tic):
        one = matrix_one
        two = matrix_two[matrix_two['Security']== tic].dropna()
        one.set_index('Date', inplace=True)
        two.set_index('Date', inplace=True)
        merge = pd.merge(one,two, how='inner', left_index=True, right_index=True)
        merge = merge.reset_index()
        merge.rename(columns={'index':'Date'}, inplace=True)
        merge['Date'] = pd.to_datetime(merge['Date'])
        return merge

    def match_matrix_FX_int_rates_diff(self,matrix_one,matrix_two,ticker_one,ticker_two):
        one = matrix_one[['Date',ticker_one]].dropna()
        two = matrix_two[['Date',ticker_two]].dropna()
        one.set_index('Date', inplace=True)
        two.set_index('Date', inplace=True)
        merge = pd.merge(one,two, how='inner', left_index=True, right_index=True)
        merge = merge.reset_index()
        merge.rename(columns={'index':'Date'}, inplace=True)
        merge['Date'] = pd.to_datetime(merge['Date'])
        return merge

    def plot_FX_series(self,mat,x_name,y_name):
        fig, ax1 = plt.subplots()
        color = 'tab:red'
        ax1.set_xlabel('Date')
        ax1.set_ylabel(x_name, color=color)
        ax1.plot(mat['Date'], mat[x_name], color=color)
        ax1.tick_params(axis='y', labelcolor=color)
        ax2 = ax1.twinx()  # instantiate a second axes that shares the same x-axis
        color = 'tab:blue'
        ax2.set_ylabel(y_name, color=color)  # we already handled the x-label with ax1
        ax2.plot(mat['Date'], mat[y_name], color=color)
        ax2.tick_params(axis='y', labelcolor=color)
        fig.tight_layout()  # otherwise the right y-label is slightly clipped
        plt.show()
    