import os
import sys
import certifi
import pandas as pd
import pymongo
from dotenv import load_dotenv
from networksecurity.exception.exception import NetworkSecurityException
from networksecurity.logging.logger import logging

load_dotenv()
# SSL সার্টিফিকেটের জন্য
ca = certifi.where()

MONGO_DB_URL = os.getenv("MONGO_DB_URL")

class NetworkDataExtract():
    def __init__(self):
        pass
        
    def cv_to_json_converter(self, file_path):
        try:
            # CSV রিড করা এবং ইনডেক্স রিসেট করা  >>>>>>>>>>>>> 'Extract' part of ETL.
            data = pd.read_csv(file_path)
            #                                   >>>>>>>>>>>>> 'Transform' part of ETL.
            data.reset_index(drop=True, inplace=True)
            
            # সরাসরি লিস্ট অফ ডিকশনারিতে রূপান্তর
            records = data.to_dict(orient='records')
            return records
        except Exception as e:
            raise NetworkSecurityException(e, sys)
    #                                           >>>>>>>>>>>>>>> 'Load' part of ETL. 
    def insert_data_to_mongoDB(self, records, database_name, collection_name):
        try:
            # ১. মঙ্গোডিবি ক্লায়েন্ট তৈরি (SSL সহ)
            mongo_client = pymongo.MongoClient(MONGO_DB_URL, tlsCAFile=ca)
            
            # ২. ডাটাবেস এবং কালেকশন সিলেক্ট করা
            database = mongo_client[database_name]
            collection = database[collection_name]
            
            # ৩. ডেটা ইনসার্ট করা
            result = collection.insert_many(records)
            
            logging.info(f"Successfully inserted {len(result.inserted_ids)} records.")
            return len(result.inserted_ids)
            
        except Exception as e:
            raise NetworkSecurityException(e, sys)

if __name__ == '__main__':
    FILE_PATH = "Network_Data/phisingData.csv"
    DATABASE = "RejwanAI"
    COLLECTION = "NetworkData"
    
    # অবজেক্ট তৈরি
    networkObj = NetworkDataExtract()
    
    # ১. কনভার্ট করা
    records = networkObj.cv_to_json_converter(file_path=FILE_PATH)
    
    # ২. ইনসার্ট করা
    no_of_records = networkObj.insert_data_to_mongoDB(records, DATABASE, COLLECTION)
    
    print(f"Total records inserted: {no_of_records}")