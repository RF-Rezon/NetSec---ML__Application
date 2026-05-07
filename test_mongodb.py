import os
from pymongo import MongoClient
from pymongo.server_api import ServerApi
from dotenv import load_dotenv


load_dotenv()

MONGO_DB_ID = os.getenv("MONGO_DB_ID")
MONGO_DB_PASS = os.getenv("MONGO_DB_PASS")


uri = f"mongodb+srv://{MONGO_DB_ID}:{MONGO_DB_PASS}@primarycluster.iog7cz5.mongodb.net/?appName=PrimaryCluster"

# Create a new client and connect to the server
client = MongoClient(uri, server_api=ServerApi('1'))

# Send a ping to confirm a successful connection
try:
    client.admin.command('ping')
    print("Pinged your deployment. You successfully connected to MongoDB!")
except Exception as e:
    print(e)