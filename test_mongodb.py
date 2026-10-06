from pymongo import MongoClient
import certifi
#password Admin123
uri =  "mongodb://prashantbabu980123_db_user:Admin123@ac-2py325z-shard-00-00.ry4iivk.mongodb.net:27017,ac-2py325z-shard-00-01.ry4iivk.mongodb.net:27017,ac-2py325z-shard-00-02.ry4iivk.mongodb.net:27017/?ssl=true&replicaSet=atlas-ct8w1h-shard-0&authSource=admin&appName=Cluster0"

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