from pymongo import MongoClient
import certifi
from typing import Dict, Tuple, List, Any
import datetime

from app.core.config import get_secret

# Global variables to store the client and collection
mongo_client = None
mongo_collection = None

def initialize_db():
    """Initialize the MongoDB connection and set global variables."""
    global mongo_client, mongo_collection
    secrets = get_secret()
    try:
        mongo_client = MongoClient(
            f"mongodb+srv://gauravadmin:{secrets['mongoDb_secret']}@cluster0.mjvbe.mongodb.net/?retryWrites=true&w=majority&appName=Cluster0",
            tls=True,
            tlsCAFile=certifi.where(),
            tlsAllowInvalidCertificates=True,
            maxPoolSize=100
        )
        db = mongo_client.se
        mongo_collection = db.fix_chunk_300tok_20overlap
        return mongo_client, mongo_collection
    except Exception as e:
        print(f"Failed to connect to MongoDB: {str(e)}")
        raise

def get_db():
    """Return the MongoDB collection."""
    return mongo_collection

def close_db_connection():
    """Close the MongoDB connection."""
    global mongo_client
    if mongo_client:
        mongo_client.close()
        print("MongoDB connection closed")

def get_mongo_fields(knumber: str, collection) -> Dict[str, str]:
    """
    Fetch the required fields from MongoDB for a given k-number.
    
    Args:
        knumber: The K-number to query
        collection: MongoDB collection object
    
    Returns:
        Dict containing the required fields with default values for missing fields
    """
    try:
        required_fields = [
            "Device Name",
            "Regulation Number",
            "Regulation Name",
            "Regulatory Class",
            "Product Code",
            "Received",
            "Device Description",
            "Operating Principle",
            "Indications for Use"
        ]

        # Query the document
        doc = collection.find_one({"K_number": knumber})
        
        if not doc:
            print(f"Warning: No document found for K-number {knumber}")
            return {field: f"K-number {knumber} Not Available" for field in required_fields}

        # Extract fields with default value for missing ones
        result = {}
        for field in required_fields:
            value = doc.get(field)
            if value is None:
                print(f"Warning: Field '{field}' missing for K-number {knumber}")
                result[field] = "Not Available"
            elif field == "Received" and isinstance(value, datetime.datetime):
                # Format datetime objects as ISO format strings
                result[field] = value.strftime("%Y-%m-%d")
            else:
                result[field] = str(value)

        return result

    except Exception as e:
        print(f"Error fetching MongoDB fields for {knumber}: {str(e)}")
        return {field: f"Error fetching MongoDB fields for {knumber}" for field in required_fields} 