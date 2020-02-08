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

class DXY():

    def __init__(self):
        pass

    def scrap(self):
        quote_page=r'https://it.investing.com/indices/usdollar-historical-data'
        headers = requests.utils.default_headers()
        headers.update({
            'User-Agent': 'Mozilla/5.0 (X11; Ubuntu; Linux x86_64; rv:52.0) Gecko/20100101 Firefox/52.0',
        })
        page=requests.get(quote_page, headers=headers)
        soup = BeautifulSoup(page.text, 'html.parser')

        tabl = soup.find('table', {"id":"curr_table"})
        listts = [TextBlob(re.sub('\n-','\nNA\n',tabl.text)).words[x:x+7] for x in range(0, len(TextBlob(re.sub('\n-','\n-\n',tabl.text)).words), 7)]
        listtslist = [list(i) for i in listts]
        dataframe = pd.DataFrame(listtslist)
        dataframe.columns=['Date','Price','Open','High','Low','Vol','Var']
        dataframe = dataframe[1:]
        dataframe = dataframe.drop(['Vol','Var'],1)
        dataframe = dataframe.dropna()
        dataframe['Open'] = [strit.replace(',','.') for strit in dataframe['Open']]
        dataframe['Price'] = [strit.replace(',','.') for strit in dataframe['Price']]
        dataframe['High'] = [strit.replace(',','.') for strit in dataframe['High']]
        dataframe['Low'] = [strit.replace(',','.') for strit in dataframe['Low']]
        dataframe.loc[:, dataframe.columns != 'Date'] = dataframe.loc[:, dataframe.columns != 'Date'].apply(pd.to_numeric, errors='coerce')
        dataframe['Date'] = pd.to_datetime(dataframe['Date'], format='%d.%m.%Y')
        return(dataframe)


    def parse(self):
        os.chdir(r'C:\Users\Federico\Documents\Trading\FX_pairs')
        DXY=pd.read_csv('DXY.csv')
        DXY = DXY[['Date','Price','High','Low','Open']]
        DXY = DXY.dropna()
        DXY.loc[:, DXY.columns != 'Date'] = DXY.loc[:, DXY.columns != 'Date'].apply(pd.to_numeric, errors='coerce')
        DXY['Date'] = pd.to_datetime(DXY['Date'])
        return(DXY)

    def merge(self,old,new):
        merg = pd.concat([new,old])
        merg = merg.sort_values('Date')
        merg = merg.drop_duplicates('Date')
        os.chdir(r'C:\Users\Federico\Documents\Trading\FX_pairs')
        merg.to_csv('DXY.csv')
        return(merg)
