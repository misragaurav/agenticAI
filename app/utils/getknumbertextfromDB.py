import pymongo
from typing import Dict, List, Any, Optional
import certifi
import datetime
import json
import time

from app.core.config import get_settings, get_secret
from app.services.mongodb import get_db

settings = get_settings()

# Configuration
CONFIG = {"max_text_length": 500000, # in units of characters, ~1MB of text ~125K tokens
          }

def escape_quotes(chunk: str) -> str:
    """
    Escape single and double quotes in a text chunk by prefixing them with backslashes.
    
    Args:
        chunk (str): The text chunk to process
        
    Returns:
        str: The chunk with escaped quotes
    """
    if not chunk:
        return chunk
        
    # Escape double quotes
    escaped = chunk.replace('"', '\\"')
    
    # Escape single quotes
    escaped = escaped.replace("'", "\\'")
    
    return escaped

def get_text_from_db(knumber: str, collection, k_number_field: str = "K_number", default_overlap: int = 100) -> str:
    """
    Retrieve and concatenate all chunks of text for a specific K-number from MongoDB.
    
    Args:
        knumber: The K-number to query
        collection: MongoDB collection object
        k_number_field: The field name in the collection that stores the K-number (default: "K_number")
        default_overlap: Default overlap to use when combining chunks
        
    Returns:
        str: Combined text from all chunks
    """
    start_time = time.time()
    
    try:
        # Query the chunks for this K-number
        query = {k_number_field: knumber}
        results = list(collection.find(query))
        
        if not results:
            print(f"No documents found for K-number {knumber}")
            return f"No text available for K-number {knumber}"
        
        # Process and combine chunks
        combined_text = ""
        overlap = default_overlap
        
        for i, chunk in enumerate(results):
            chunk_text = chunk.get("chunks", "")
            
            if i == 0:
                # First chunk - use entire text
                combined_text = chunk_text
            else:
                # Find overlap with previous chunk to avoid duplication
                if overlap > 0 and len(chunk_text) > overlap:
                    combined_text += chunk_text[overlap:]
                else:
                    combined_text += chunk_text
        
        # Truncate if too long
        if len(combined_text) > CONFIG["max_text_length"]:
            combined_text = combined_text[:CONFIG["max_text_length"]] + "... [text truncated]"
        
        end_time = time.time()
        print(f"Retrieved text for {knumber} in {end_time - start_time:.2f} seconds")
        
        return combined_text
        
    except Exception as e:
        print(f"Error retrieving text for {knumber}: {str(e)}")
        return f"Error retrieving text: {str(e)}"
