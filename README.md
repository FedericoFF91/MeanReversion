# MeanReversion
Contains mean reversion algo on FX written in Python

-- MeanRevertFX : It contains the class with the different methods (I should optimize this class better, some parameters are still coded as static inside the methods, this should be the next step)

-- MeanRevertFXBacktest : It is using the class defined in the MeanRevertFX script in order to run a backtesting (data is not provided, I will do it as soon as possible)

-- ParseMergeFXSeries : It is a class which contains some of the methods used in the backtest to handle the FX time series (cleaning and Merging)

-- MongoClass : It is a wrapper class around pymongo library which I use in order to store data in my personal mongo database

-- PlottingIntDiffFXPairs: It is a class which contains methods to plot FX and interest rates time series used in the Backtest script

-- DXY: it is a class which scrapes and save data for the DXY index (I should probably merge this residual class inside the the ParseMergeFXSeries, I will do it as soon as I have some time). It's used in the backtest script

-- Johansen : It is a class which contains methods for johansen test (I took it somewhere on Git, it is not my work). It is used in the Backtest script