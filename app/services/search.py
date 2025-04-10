from typing import List, Dict, Any, Tuple
from pymongo.command_cursor import CommandCursor
import pandas as pd
from app.utils.embeddings import get_embedding_from_bedrock
from app.core.config import get_settings, KEYWORD_EXTRACTION_COUNT, SYNONYM_COUNT_PER_KEYWORD, SEARCH_WEIGHTS_TUPLE
from app.services.llm import LLMClient
from app.utils.getknumbertextfromDB import escape_quotes
import json
import asyncio # Import asyncio

settings = get_settings()

def query_mongo_db(collection, query: str, model_id=None, dimensions=None, normalize=True) -> CommandCursor[dict]:
    """
    Search MongoDB using vector search with embeddings.
    
    Args:
        collection: MongoDB collection to search
        query: Text query to search for
        model_id: Model ID to use for embeddings
        dimensions: Embedding dimensions
        normalize: Whether to normalize embeddings
    
    Returns:
        MongoDB command cursor with search results
    """
    # Field names in DB
    knumber_field_name = "metadata"
    text_field_name = "chunks"
    text_field_embed_name = "titanv2_embedding"
    #index_field_name_1 = "vector_index"
    index_field_name_1="vector_index_quantized_8bit"

    print("Vectorizing the query")
    embedding = get_embedding_from_bedrock(query, model_id=model_id, dimensions=dimensions, normalize=normalize)
    print("Done Vectorizing the query")

    # Number of nearest neighbors to use during the search
    num_of_NN = 70
    # Number of documents to return
    num_of_results_to_return = 70
    # ENN or ANN
    is_ENN = False
    
    print("Searching MongoDB")

    if not is_ENN:
        results = collection.aggregate([
        {"$vectorSearch": {
            "exact": is_ENN,
            "queryVector": embedding,
            "path": text_field_embed_name,
            "numCandidates": num_of_NN,
            "limit": num_of_results_to_return,
            "index": index_field_name_1,
                }
            },
            {'$project': {'_id': 0,
            knumber_field_name: 1,
            'score': {'$meta': 'vectorSearchScore'}
            }
            }
        ])

    if is_ENN:
        results = collection.aggregate([
        {"$vectorSearch": {
            "exact": is_ENN,
            "queryVector": embedding,
            "path": text_field_embed_name,
            "limit": num_of_results_to_return,
            "index": index_field_name_1,
                }
            },
            {'$project': {'_id': 0,
            knumber_field_name: 1,
            'score': {'$meta': 'vectorSearchScore'}
            }
            }
        ])

    print("Done Searching MongoDB")
    return results

def rem_acronyms(text: str) -> str:
    """
    Remove acronyms (uppercase words with 2-5 letters) from text.
    
    Args:
        text: Input text
        
    Returns:
        Text with acronyms removed
    """
    import re
    # This regex looks for uppercase words with 2-5 letters that may include digits
    pattern = r'\b[A-Z][A-Z0-9]{1,4}\b'
    # Replace acronyms with empty string
    return re.sub(pattern, '', text)


def enrich_query_with_keywords(query: str, keywords_synonyms: Dict[str, List[str]]) -> str:
    """
    Enriches a query string by appending keywords and their synonyms.

    Args:
        query: The original query text.
        keywords_synonyms: Dictionary where keys are keywords and values are lists of synonyms.

    Returns:
        The enriched query string.
    """
    enriched_query = query
    for keyword, synonyms in keywords_synonyms.items():
        # Add keyword if not already in the query (case-insensitive check)
        if keyword.lower() not in enriched_query.lower():
            enriched_query += f" {keyword}"
        # Add synonyms if not already in the query (case-insensitive check)
        for synonym in synonyms:
            if synonym.lower() not in enriched_query.lower():
                enriched_query += f" {synonym}"
    return enriched_query


def vector_search(
    collection,
    query: str, # This query should be the final query string (potentially enriched)
    knumber_field_name: str = "K_number",
    text_field_name: str = "chunks",
    text_field_embed_name: str = "titanv2_embedding",
    index_field_name_1: str = "vector_index_quantized_8bit",
    model_id: str = None,
    dimensions: int = None,
    normalize: bool = True,
    num_of_NN: int = 70,
    num_of_results_to_return: int = 70,
    is_ENN: bool = False
) -> CommandCursor[dict]:
    """
    Performs vector search on a MongoDB collection using a given query string.

    Args:
        collection: MongoDB collection
        query: The query text to search for (should be pre-processed/enriched if needed).
        knumber_field_name: Field name for K-number
        text_field_name: Field name for text
        text_field_embed_name: Field name for embedding
        index_field_name_1: Field name for vector search index
        model_id: Model ID for embeddings
        dimensions: Embedding dimensions
        normalize: Whether to normalize embeddings
        num_of_NN: Number of nearest neighbors for ANN search
        num_of_results_to_return: Maximum results to return
        is_ENN: Whether to use exact nearest neighbors (ENN) search.

    Returns:
        MongoDB command cursor with search results
    """
    # Query enrichment logic removed - expects 'query' to be ready

    # Use default settings if not provided
    model_id = model_id or settings.EMBEDDING_MODEL_ID
    dimensions = dimensions or settings.EMBEDDING_DIMENSIONS

    print(f"Vectorizing query: {query[:100]}...")
    embedding = get_embedding_from_bedrock(query, model_id=model_id, dimensions=dimensions, normalize=normalize)
    print("Done Vectorizing the query")

    print("Searching MongoDB with query")

    # Define the common $vectorSearch stage structure
    vector_search_stage = {
        "$vectorSearch": {
            "queryVector": embedding,
            "path": text_field_embed_name,
            "limit": num_of_results_to_return,
            "index": index_field_name_1,
        }
    }

    # Add specific parameters based on ENN or ANN
    if is_ENN:
        vector_search_stage["$vectorSearch"]["exact"] = True
    else:
        vector_search_stage["$vectorSearch"]["exact"] = False
        vector_search_stage["$vectorSearch"]["numCandidates"] = num_of_NN

    # Define the common $project stage
    project_stage = {
        '$project': {
            '_id': 0,
            knumber_field_name: 1,
            'score': {'$meta': 'vectorSearchScore'}
        }
    }

    # Execute the aggregation pipeline
    results = collection.aggregate([vector_search_stage, project_stage])

    print("Done Searching MongoDB")
    return results


def combine_3_search_results(
    df1: pd.DataFrame,  # Indications results
    df2: pd.DataFrame,  # Device details results
    df3: pd.DataFrame,  # Operating principle results
    weights: Tuple[float, float, float] = SEARCH_WEIGHTS_TUPLE  # Default weights from config
) -> List[Tuple[str, float]]:
    """
    Combine search results from three different searches and rank by weighted total score.
    
    Args:
        df1: Indications search results DataFrame with 'knumber' and 'score' columns
        df2: Device details search results DataFrame with 'knumber' and 'score' columns
        df3: Operating principle search results DataFrame with 'knumber' and 'score' columns
        weights: Tuple of (indications_weight, device_details_weight, operating_principle_weight)
                 Default weights prioritize indications (0.5) over device details (0.3) and operating principle (0.2)
        
    Returns:
        List of tuples (knumber, total_score) sorted by total_score in descending order
    """
    # Validate weights
    if sum(weights) != 1.0:
        print(f"Warning: Weights {weights} don't sum to 1.0. Normalizing...")
        total = sum(weights)
        weights = tuple(w/total for w in weights)
        print(f"Normalized weights: {weights}")
    
    indications_weight, device_details_weight, operating_principle_weight = weights
    
    # Merge the three dataframes on knumber
    merged_df = pd.merge(df1, df2, on='knumber', how='outer', suffixes=('_1', '_2'))
    merged_df = pd.merge(merged_df, df3, on='knumber', how='outer')
    
    # Rename the third score column
    merged_df.rename(columns={'score': 'score_3'}, inplace=True)
    
    # Replace NaN with 0 for knumbers that weren't found in all searches
    merged_df.fillna(0, inplace=True)
    
    # Calculate weighted total score
    merged_df['total_score'] = (
        indications_weight * merged_df['score_1'] + 
        device_details_weight * merged_df['score_2'] + 
        operating_principle_weight * merged_df['score_3']
    )
    
    # Add component scores for transparency
    merged_df['indications_contribution'] = indications_weight * merged_df['score_1']
    merged_df['device_details_contribution'] = device_details_weight * merged_df['score_2']
    merged_df['operating_principle_contribution'] = operating_principle_weight * merged_df['score_3']
    
    # Print scoring information
    print(f"\nScoring information:")
    print(f"- Indications weight: {indications_weight:.2f}")
    print(f"- Device details weight: {device_details_weight:.2f}")
    print(f"- Operating principle weight: {operating_principle_weight:.2f}")
    
    # Sort by total score in descending order
    sorted_df = merged_df.sort_values('total_score', ascending=False)
    
    # For the top 5 results, print the breakdown of scores
    print("\nScore breakdown for top 5 results:")
    for i, row in sorted_df.head(5).iterrows():
        print(f"K-number: {row['knumber']}, Total score: {row['total_score']:.3f}")
        print(f"  - Indications: {row['score_1']:.3f} × {indications_weight:.2f} = {row['indications_contribution']:.3f}")
        print(f"  - Device details: {row['score_2']:.3f} × {device_details_weight:.2f} = {row['device_details_contribution']:.3f}")
        print(f"  - Operating principle: {row['score_3']:.3f} × {operating_principle_weight:.2f} = {row['operating_principle_contribution']:.3f}")
    
    # Convert to list of tuples
    result = [(row['knumber'], float(round(row['total_score'], 3))) for _, row in sorted_df.iterrows()]
    
    return result


# Make the function async
async def query_mongo_LLM_assist(
    collection,
    enriched_indications_query: str,
    enriched_device_details_query: str,
    enriched_operating_principle_query: str,
    # Keep other parameters as they relate to the search execution
    knumber_field_name: str = "K_number",
    text_field_name: str = "chunks",
    text_field_embed_name: str = "titanv2_embedding",
    index_field_name_1: str = "vector_index_quantized_8bit",
    model_id: str = None,
    dimensions: int = None,
    normalize: bool = True,
    num_of_NN: int = 70,
    num_of_results_to_return: int = 70,
    is_ENN: bool = False,
    weights: Tuple[float, float, float] = SEARCH_WEIGHTS_TUPLE
) -> List[Tuple[str, float]]:
    """
    Performs multiple vector searches CONCURRENTLY using pre-enriched queries and combines results.
    (Runs synchronous vector_search calls in separate threads via asyncio executor)

    Args:
        collection: MongoDB collection
        enriched_indications_query: Enriched query string for indications.
        enriched_device_details_query: Enriched query string for device details.
        enriched_operating_principle_query: Enriched query string for operating principle.
        knumber_field_name: Field name for K-number
        text_field_name: Field name for text (potentially less relevant now)
        text_field_embed_name: Field name for embedding
        index_field_name_1: Field name for vector search index
        model_id: Model ID for embeddings
        dimensions: Embedding dimensions
        normalize: Whether to normalize embeddings
        num_of_NN: Number of nearest neighbors
        num_of_results_to_return: Maximum results to return
        is_ENN: Whether to use exact nearest neighbors
        weights: Tuple of (indications_weight, device_details_weight, operating_principle_weight)

    Returns:
        List of tuples (knumber, total_score) sorted by total_score in descending order
    """
    print("Starting CONCURRENT multi-query MongoDB search with pre-enriched queries")
    loop = asyncio.get_event_loop()

    # --- Prepare arguments for each vector_search call --- 
    common_args = {
        "collection": collection,
        "knumber_field_name": knumber_field_name,
        "text_field_name": text_field_name,
        "text_field_embed_name": text_field_embed_name,
        "index_field_name_1": index_field_name_1,
        "model_id": model_id,
        "dimensions": dimensions,
        "normalize": normalize,
        "num_of_NN": num_of_NN,
        "num_of_results_to_return": num_of_results_to_return,
        "is_ENN": is_ENN
    }

    indications_args = {"query": enriched_indications_query, **common_args}
    device_details_args = {"query": enriched_device_details_query, **common_args}
    operating_principle_args = {"query": enriched_operating_principle_query, **common_args}

    # --- Run Vector Searches Concurrently using Executor --- 
    print("Scheduling vector searches concurrently...")
    
    # Use lambda to wrap the function call with its specific arguments
    indications_task = loop.run_in_executor(None, lambda: vector_search(**indications_args))
    device_details_task = loop.run_in_executor(None, lambda: vector_search(**device_details_args))
    operating_principle_task = loop.run_in_executor(None, lambda: vector_search(**operating_principle_args))

    # --- Gather Results --- 
    results = await asyncio.gather(
        indications_task,
        device_details_task,
        operating_principle_task
    )
    
    indications_results = results[0]
    device_details_results = results[1]
    operating_principle_results = results[2]
    print("Concurrent vector searches completed.")

    # --- Process and Combine Results (remains synchronous) --- 
    print("Processing results...")
    # Convert MongoDB cursors to DataFrames
    indications_df = pd.DataFrame(list(indications_results)) 
    device_details_df = pd.DataFrame(list(device_details_results))
    operating_principle_df = pd.DataFrame(list(operating_principle_results))
    
    # Handle cases where a search might return no results
    if indications_df.empty:
        indications_df = pd.DataFrame(columns=[knumber_field_name, 'score'])
    if device_details_df.empty:
        device_details_df = pd.DataFrame(columns=[knumber_field_name, 'score'])
    if operating_principle_df.empty:
        operating_principle_df = pd.DataFrame(columns=[knumber_field_name, 'score'])

    # Group by K-number and get max score for each search
    indications_grouped = indications_df.groupby(knumber_field_name)['score'].max().reset_index()
    indications_grouped.rename(columns={knumber_field_name: 'knumber'}, inplace=True)
    
    device_details_grouped = device_details_df.groupby(knumber_field_name)['score'].max().reset_index()
    device_details_grouped.rename(columns={knumber_field_name: 'knumber'}, inplace=True)
    
    operating_principle_grouped = operating_principle_df.groupby(knumber_field_name)['score'].max().reset_index()
    operating_principle_grouped.rename(columns={knumber_field_name: 'knumber'}, inplace=True)
    
    # Combine results using the existing synchronous function
    print("Combining search results...")
    combined_results = combine_3_search_results(
        indications_grouped,
        device_details_grouped,
        operating_principle_grouped,
        weights
    )
    
    print(f"Found {len(combined_results)} combined results")
    return combined_results 
