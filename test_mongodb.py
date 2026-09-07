from pymongo import MongoClient
import certifi

uri =  "mongodb+srv://prashantbabu980123_db_user:Admin123@cluster0.ry4iivk.mongodb.net/?appName=Cluster0"

print("Testing MongoDB connection...")

client = MongoClient(
    uri,
    tls=True,
    tlsCAFile=certifi.where(),
    serverSelectionTimeoutMS=10000,
    connectTimeoutMS=10000
)

try:
    print(client.admin.command("ping"))
    print("MongoDB connected successfully!")
except Exception as e:
    print(repr(e))