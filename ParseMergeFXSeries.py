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
from selenium import webdriver
import win32clipboard
import argparse
from MongoClass import MongoDataHandling
import pyautogui

class ParseReadFxSeries(MongoDataHandling):

    def __init__(self):
        pass

    @classmethod
    def GetClipboardData(self):
        """
        Extracts the data currently saved in the clipboard and returns it in string format.

        Arguments:

        Returns:
        clipboard_data -- The data contained in the clipboard in string format
        """
        win32clipboard.OpenClipboard()
        clipboard_data = win32clipboard.GetClipboardData()
        win32clipboard.CloseClipboard()
        return(clipboard_data)
    
    @classmethod
    def DownloadFullHistoryFXDataSeries(self,BaseCurrency,SecondCurrency,Update):

        # define functions to click and pause
        def pause(wait = 2):
            sleep(wait)

        def click():
            m.press(Button.left)
            pause()
            m.release(Button.left)

        def rclick():
            m.press(Button.right)
            pause()
            m.release(Button.right)
        
        m = mouse.Controller()
        Button = mouse.Button
        k = keyboard.Controller()
        Key = keyboard.Key

        # open browser and get the page, click on space in order to move down and 
        # visualize the object
        driver = webdriver.Firefox()
        PageUrl = r'https://it.investing.com/currencies/' + str(BaseCurrency) +r'-'+str(SecondCurrency)+r'-historical-data#datePickerIconWrap'
        driver.get(PageUrl)
        pause(2)
        # move down 
        k.press(Key.space)
        k.release(Key.space)
        pause(2)
        # click on the DatePicker table in order
        # to open the tables to select the dates
        ClickPosition = (825, 341) 
        m.position = ClickPosition
        pause(1)
        click()
        # write the date into the start date DatePicker
        # do not need to write all the date, only the year that we want
        # in this case 2000 because all the time series will start from there
        pause(2)
        k.type('2000')
        pause()
        k.press(Key.enter)
        # click on Apply in order to made the changes
        ApplyPosition = (800,551)
        m.position = ApplyPosition
        pause(2)
        click()
        # Select and Copy to Clipboard
        # Used this way instead of use Beautiful Soup
        # because it did not work properly (it requires too much time to write that correctly)
        # so I decided to simply select everything and right click and copy to clipboard
        # after I will get only the historical table copied to clipboard
        pause(10)
        k.press(Key.ctrl)
        k.press("a")
        k.release(Key.ctrl)
        k.release("a")
        pause()
        CenterPage = (562,518)
        m.position = CenterPage
        rclick()
        pause()
        CopyPosition = (621,536)
        m.position = CopyPosition
        click()
        pause()
        ClipboardData = self.GetClipboardData()
        # Convert the Data copied into the clipboard to TextBlob
        FXTimeSeriesText = TextBlob(ClipboardData)
        # find the position of the first word in the table
        FirstWord = FXTimeSeriesText.find("Data")
        # find the position of the last word
        LastWord = FXTimeSeriesText.find("Media") - 55
        # select the table
        FXTable = FXTimeSeriesText[FirstWord:LastWord]
        # convert to a list
        ListFXTable = []
        for i in range(0,len(FXTable.words),6):
            ListFXTable.append(list(FXTable.words[i:(i+5)]))
        # convert to a PandaDataframe
        FXTab = pd.DataFrame(ListFXTable)
        # rename columns
        FXTab.columns = FXTab.iloc[0]
        FXTab = FXTab[1:]
        # close the browser
        driver.close()
        # format the dataset
        FXTab.columns = ['Date','Price','Open','Max','Min']
        FXTab['Date'] = pd.to_datetime(FXTab['Date'], format="%d.%m.%Y")
        FXTab['Price'] = pd.to_numeric(FXTab['Price'].str.replace(',','.'))
        FXTab['Open'] = pd.to_numeric(FXTab['Open'].str.replace(',','.'))
        FXTab['Max'] = pd.to_numeric(FXTab['Max'].str.replace(',','.'))
        FXTab['Min'] = pd.to_numeric(FXTab['Min'].str.replace(',','.'))
        # Save into the Database ('Same collection')
        if Update.lower() in ('true','yes','t','y'):
            Tab = self.UpdateTimeSeries(FXTab,'MarketData',str(BaseCurrency).upper() +str(SecondCurrency).upper())
        elif Update.lower() in ('false','no','f','n'):
            Tab = self.UploadTimeSeries(FXTab,'MarketData',str(BaseCurrency).upper() +str(SecondCurrency).upper())
        else:
            raise argparse.ArgumentTypeError('Boolean Value Expected with '' ')            
        return Tab

    @classmethod
    def UpdateFXDataSeries(self,BaseCurrency,SecondCurrency,DateToExtract):
        # define functions to click and pause
        def pause(wait = 2):
            sleep(wait)

        def click():
            m.press(Button.left)
            pause()
            m.release(Button.left)

        def rclick():
            m.press(Button.right)
            pause()
            m.release(Button.right)

        m = mouse.Controller()
        Button = mouse.Button
        k = keyboard.Controller()
        Key = keyboard.Key

        # open browser and get the page, click on space in order to move down and 
        # visualize the object
        driver = webdriver.Firefox()
        PageUrl = r'https://it.investing.com/currencies/' + str(BaseCurrency) +r'-'+str(SecondCurrency)+r'-historical-data#datePickerIconWrap'
        driver.get(PageUrl)
        pause(2)
        # move down 
        k.press(Key.space)
        k.release(Key.space)
        pause(2)
        # click on the DatePicker table in order
        # to open the tables to select the dates
        ClickPosition = (825, 341) 
        m.position = ClickPosition
        pause(1)
        click()
        # write the date into the start date DatePicker
        # in this case you need to provide the date that
        # you want to use (DateToExtract) because all the time series will start from there
        pause(2)
        k.type(DateToExtract)
        pause()
        k.press(Key.enter)
        # click on Apply in order to made the changes
        ApplyPosition = (800,551)
        m.position = ApplyPosition
        pause(2)
        click()
        # Select and Copy to Clipboard
        # Used this way instead of use Beautiful Soup
        # because it did not work properly (it requires too much time to write that correctly)
        # so I decided to simply select everything and right click and copy to clipboard
        # after I will get only the historical table copied to clipboard
        pause(10)
        k.press(Key.ctrl)
        k.press("a")
        k.release(Key.ctrl)
        k.release("a")
        pause()
        CenterPage = (562,518)
        m.position = CenterPage
        rclick()
        pause()
        CopyPosition = (621,536)
        m.position = CopyPosition
        click()
        pause()
        ClipboardData = self.GetClipboardData()
        # Convert the Data copied into the clipboard to TextBlob
        FXTimeSeriesText = TextBlob(ClipboardData)
        # find the position of the first word in the table
        FirstWord = FXTimeSeriesText.find("Data")
        # find the position of the last word
        LastWord = FXTimeSeriesText.find("Media") - 55
        # select the table
        FXTable = FXTimeSeriesText[FirstWord:LastWord]
        # convert to a list
        ListFXTable = []
        for i in range(0,len(FXTable.words),6):
            ListFXTable.append(list(FXTable.words[i:(i+5)]))
        # convert to a PandaDataframe
        FXTab = pd.DataFrame(ListFXTable)
        # rename columns
        FXTab.columns = FXTab.iloc[0]
        FXTab = FXTab[1:]
        # close the browser
        driver.close()
        # format the dataset
        FXTab.columns = ['Date','Price','Open','Max','Min']
        FXTab['Date'] = pd.to_datetime(FXTab['Date'], format="%d.%m.%Y")
        FXTab['Price'] = pd.to_numeric(FXTab['Price'].str.replace(',','.'))
        FXTab['Open'] = pd.to_numeric(FXTab['Open'].str.replace(',','.'))
        FXTab['Max'] = pd.to_numeric(FXTab['Max'].str.replace(',','.'))
        FXTab['Min'] = pd.to_numeric(FXTab['Min'].str.replace(',','.'))
        # Upload Collection on the Database ('Same collection')
        FxTabHistoric = self.ReadDataToPanda('MarketData',str(BaseCurrency).upper() +str(SecondCurrency).upper())
        # merge the two tables together
        FXTabFinal = pd.concat([FXTab,FxTabHistoric])
        # drop duplicates
        FXTabFinal = FXTabFinal.drop_duplicates()
        # Save into the Database ('Same collection')
        Tab = self.UpdateTimeSeries(FXTabFinal,'MarketData',str(BaseCurrency).upper() +str(SecondCurrency).upper())
        return Tab

    def ScrapeFXtimeseries(self,BaseCurrency,SecondCurrency):    
        headers = requests.utils.default_headers()
        headers.update({
            'User-Agent': 'Mozilla/5.0 (X11; Ubuntu; Linux x86_64; rv:52.0) Gecko/20100101 Firefox/52.0',
        })
        page=requests.get(PageUrl, headers=headers)
        soup = BeautifulSoup(page.text, 'html.parser')

        tabl = soup.find('table', {"id":"curr_table"})
        listts = [TextBlob(tabl.text).words[x:x+6] for x in range(0, len(TextBlob(tabl.text).words), 6)]
        listtslist = [list(i) for i in listts]
        dataframe = pd.DataFrame(listtslist)
        dataframe.columns=['Date','Price','Open','High','Low','Var']
        dataframe = dataframe[1:]
        dataframe = dataframe.drop(['Var'],1)
        dataframe = dataframe.dropna()
        dataframe['Open'] = [strit.replace(',','.') for strit in dataframe['Open']]
        dataframe['Price'] = [strit.replace(',','.') for strit in dataframe['Price']]
        dataframe['High'] = [strit.replace(',','.') for strit in dataframe['High']]
        dataframe['Low'] = [strit.replace(',','.') for strit in dataframe['Low']]
        dataframe.loc[:, dataframe.columns != 'Date'] = dataframe.loc[:, dataframe.columns != 'Date'].apply(pd.to_numeric, errors='coerce')
        dataframe['Date'] = pd.to_datetime(dataframe['Date'], format='%d.%m.%Y')
        return(dataframe)


    def parse(self,filename):
        os.chdir(r'C:\Users\Federico\Documents\Trading\FX_pairs')
        s = [file for file in os.listdir(r'C:\Users\Federico\Documents\Trading\FX_pairs') if re.match(filename,file)]
        DXY=pd.read_csv(s[0])
        DXY = DXY.dropna()
        DXY.loc[:, DXY.columns != 'Date'] = DXY.loc[:, DXY.columns != 'Date'].apply(pd.to_numeric, errors='coerce')
        DXY['Date'] = pd.to_datetime(DXY['Date'])
        return(DXY)

    def merge(self,old,new,filename):
        old = old[['Date','High','Low','Open','Price']]
        merg = pd.concat([new,old])
        merg = merg.sort_values('Date')
        merg = merg.drop_duplicates('Date')
        os.chdir(r'C:\Users\Federico\Documents\Trading\FX_pairs')
        merg.to_csv(filename)
        return(merg)


