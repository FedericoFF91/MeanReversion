import pymongo
from pymongo import MongoClient
import pandas as pd

# Class wrapper to handle Mongo import and export
class MongoDataHandling(object):

    def __init__(self):
        pass

    @classmethod
    def UpdateTimeSeries(self,Df,Db,Coll):
        # Open connection with client
        Client = MongoClient()
        # select the database and collection that we want
        Collection = Client[Db][Coll]
        # format data as dictionary
        # every line is a different document
        Data = Df.to_dict(orient='records')
        # Rename the collection from the database
        # and after that remove it if the new is created correctly
        # insert documents into a new collection
        # (For now this is working because when you scrape
        # you get everything but it needs to be improved a lot)
        Collection.rename(Coll+'_test')
        Collection.insert_many(Data)
        Client[Db][Coll+'_test'].drop()
        return Df   

    @classmethod
    def UploadTimeSeries(self,Df,Db,Coll):
        # Open connection with client
        Client = MongoClient()
        # select the database and collection that we want
        Collection = Client[Db][Coll]
        # format data as dictionary
        # every line is a different document
        Data = Df.to_dict(orient='records')
        # insert documents into a new collection
        # (For now this is working because when you scrape
        # you get everything but it needs to be improved a lot)
        Collection.insert_many(Data)
        return Df

    @classmethod
    def ReadDataToPanda(self,Db,Coll):
        # select the collection that we want
        Client = MongoClient()
        Database = Client[Db]
        collection = Database[Coll]
        # save data into a pandas
        data = pd.DataFrame(list(collection.find()))
        # remove _id from the columns
        data = data.drop('_id',1)
        return data

