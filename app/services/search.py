from typing import List, Dict, Any, Tuple
from pymongo.command_cursor import CommandCursor
import pandas as pd
from app.utils.embeddings import get_embedding_from_bedrock
from app.core.config import get_settings, KEYWORD_EXTRACTION_COUNT, SYNONYM_COUNT_PER_KEYWORD, SEARCH_WEIGHTS_TUPLE
from app.services.llm import LLMClient
from app.utils.getknumbertextfromDB import escape_quotes
import json

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
    index_field_name_1 = "vector_index"

    print("Vectorizing the query")
    embedding = get_embedding_from_bedrock(query, model_id=model_id, dimensions=dimensions, normalize=normalize)
    print("Done Vectorizing the query")

    # Number of nearest neighbors to use during the search
    num_of_NN = 10000
    # Number of documents to return
    num_of_results_to_return = 100
    # ENN or ANN
    is_ENN = False
    
    print("Searching MongoDB")

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

    print("Done Searching MongoDB")
    return results


def get_keywords_synonyms(text: str, num_keywords: int = KEYWORD_EXTRACTION_COUNT, num_synonyms: int = SYNONYM_COUNT_PER_KEYWORD) -> Dict[str, List[str]]:
    """
    Extract keywords from text and generate synonyms for each keyword using LLM.
    
    Args:
        text: Input text
        num_keywords: Number of keywords to extract (default from config)
        num_synonyms: Number of synonyms to generate per keyword (default from config)
        
    Returns:
        Dictionary with keywords as keys and lists of synonyms as values
    """
    llm_client = LLMClient(settings.LLM_MODEL)
    
    prompt = f"""
    You are tasked with extracting key medical device terminology from the following text and generating relevant synonyms.
    
    Text:
    {text}
    
    Please extract exactly {num_keywords} important keywords from the text. For each keyword, provide {num_synonyms} synonyms or related terms.
    
    Return your response ONLY as a JSON object where each key is a keyword and its value is an array of synonyms. For example:
    {{
        "imaging device": ["radiological system", "medical imaging apparatus", "diagnostic imaging unit"],
        "radiography": ["X-ray imaging", "radiological examination", "radiographic procedure"]
    }}
    
    Focus on technical terms, medical device features, and specific functions mentioned in the text.
    """
    
    try:
        # First attempt with JSON response format
        response = llm_client.complete(
            prompt, 
            max_tokens=1000, 
            temperature=0.2,
            top_p=0.95,
            stream=False,
            response_format={"type": "json_object"},
            extra_body={"service_tier": "on_demand"}
        )
        
        # Parse the JSON response
        import json
        try:
            result = json.loads(response)
            
            # Verify we have enough keywords
            if len(result) < num_keywords:
                print(f"Warning: Only extracted {len(result)} keywords, needed {num_keywords}")
                
            return result
        except json.JSONDecodeError as json_err:
            print(f"Failed to parse JSON response: {str(json_err)}")
            print(f"Response received: {response[:100]}...")
            # Continue to fallback methods
        
    except Exception as e:
        print(f"JSON generation failed: {str(e)}")
        
        # Fallback to plain text format
        try:
            fallback_prompt = f"""
            You are tasked with extracting key medical device terminology from the following text and generating relevant synonyms.
            
            Text:
            {text}
            
            Please extract exactly {num_keywords} important keywords from the text. For each keyword, provide {num_synonyms} synonyms or related terms.
            
            Format your response as follows:
            
            KEYWORD: keyword1
            SYNONYMS: synonym1, synonym2, synonym3
            
            KEYWORD: keyword2
            SYNONYMS: synonym1, synonym2, synonym3
            
            Focus on technical terms, medical device features, and specific functions mentioned in the text.
            """
            
            response = llm_client.complete(
                fallback_prompt, 
                max_tokens=1000, 
                temperature=0.2,
                top_p=0.95,
                stream=False,
                response_format=None,  # No JSON
                extra_body={"service_tier": "on_demand"}
            )
            
            # Parse line-by-line format
            result = {}
            lines = response.strip().split('\n')
            
            current_keyword = None
            
            for line in lines:
                line = line.strip()
                if not line:
                    continue
                    
                if line.upper().startswith('KEYWORD:'):
                    current_keyword = line[line.find(':')+1:].strip()
                elif (line.upper().startswith('SYNONYMS:') or line.upper().startswith('SYNONYM:')) and current_keyword:
                    synonyms_part = line[line.find(':')+1:].strip()
                    synonyms = [s.strip() for s in synonyms_part.split(',')]
                    while len(synonyms) < num_synonyms:
                        synonyms.append(f"variant of {current_keyword}")
                    
                    result[current_keyword] = synonyms[:num_synonyms]
                    current_keyword = None
            
            if result:
                return result
        except Exception as nested_e:
            print(f"\n\n Fallback parsing failed: {str(nested_e)}")
        
        # Final fallback: return generic terms
        print("Using blank as fallback")
        fallback_keywords = [
            " "
        ]
        
        result = {}
        for i in range(min(num_keywords, len(fallback_keywords))):
            keyword = fallback_keywords[i]
            result[keyword] = [f"{keyword} variant", f"alternative {keyword}", f"similar {keyword}"][:num_synonyms]
            
        return result

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


def vector_search(
    collection, 
    query: str, 
    keywords_synonyms: Dict[str, List[str]],
    knumber_field_name: str = "K_number",
    text_field_name: str = "chunks",
    text_field_embed_name: str = "titanv2_embedding",
    index_field_name_1: str = "vector_index",
    model_id: str = None, 
    dimensions: int = None, 
    normalize: bool = True,
    num_of_NN: int = 10000,
    num_of_results_to_return: int = 100,
    is_ENN: bool = False
) -> CommandCursor[dict]:
    """
    Enhanced semantic search using query enriched with keywords and synonyms.
    
    Args:
        collection: MongoDB collection
        query: Original query text
        keywords_synonyms: Dictionary of keywords and their synonyms
        knumber_field_name: Field name for K-number
        text_field_name: Field name for text
        text_field_embed_name: Field name for embedding
        index_field_name_1: Field name for vector search index
        model_id: Model ID for embeddings
        dimensions: Embedding dimensions
        normalize: Whether to normalize embeddings
        num_of_NN: Number of nearest neighbors
        num_of_results_to_return: Maximum results to return
        is_ENN: Whether to use exact nearest neighbors
        
    Returns:
        MongoDB command cursor with search results
    """
    # Enrich query with keywords and synonyms
    enhanced_query = query
    
    for keyword, synonyms in keywords_synonyms.items():
        if keyword.lower() not in query.lower():
            enhanced_query += f" {keyword}"
        for synonym in synonyms:
            if synonym.lower() not in query.lower() and synonym.lower() not in enhanced_query.lower():
                enhanced_query += f" {synonym}"
    
    # Use default settings if not provided
    model_id = model_id or settings.BEDROCK_MODEL_ID
    dimensions = dimensions or settings.EMBEDDING_DIMENSIONS
    
    print(f"Enhanced query: {enhanced_query[:100]}...")
    print("Vectorizing the enhanced query")
    embedding = get_embedding_from_bedrock(enhanced_query, model_id=model_id, dimensions=dimensions, normalize=normalize)
    print("Done Vectorizing the enhanced query")
    
    print("Searching MongoDB with enhanced query")
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
    
    print("Done Searching MongoDB with enhanced query")
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


def query_mongo_LLM_assist(
    collection, 
    indications: str,
    device_details: str,
    operating_principle: str,
    knumber_field_name: str = "K_number",
    text_field_name: str = "chunks",
    text_field_embed_name: str = "titanv2_embedding",
    index_field_name_1: str = "vector_index",
    model_id: str = None, 
    dimensions: int = None, 
    normalize: bool = True,
    num_of_NN: int = 10000,
    num_of_results_to_return: int = 100,
    is_ENN: bool = False,
    num_keywords: int = KEYWORD_EXTRACTION_COUNT,
    num_synonyms: int = SYNONYM_COUNT_PER_KEYWORD,
    remove_acronyms: bool = False,
    weights: Tuple[float, float, float] = SEARCH_WEIGHTS_TUPLE  # Default weights from config
) -> List[Tuple[str, float]]:
    """
    Advanced MongoDB search using LLM-assisted keyword extraction and synonym generation.
    
    Args:
        collection: MongoDB collection
        indications: Indications for use text
        device_details: Device details text
        operating_principle: Operating principle text
        knumber_field_name: Field name for K-number
        text_field_name: Field name for text
        text_field_embed_name: Field name for embedding
        index_field_name_1: Field name for vector search index
        model_id: Model ID for embeddings
        dimensions: Embedding dimensions
        normalize: Whether to normalize embeddings
        num_of_NN: Number of nearest neighbors
        num_of_results_to_return: Maximum results to return
        is_ENN: Whether to use exact nearest neighbors
        num_keywords: Number of keywords to extract
        num_synonyms: Number of synonyms per keyword
        remove_acronyms: Whether to remove acronyms from input texts
        weights: Tuple of (indications_weight, device_details_weight, operating_principle_weight)
                 Default weights prioritize indications (0.5) over device details (0.3) and operating principle (0.2)
        
    Returns:
        List of tuples (knumber, total_score) sorted by total_score in descending order
    """
    print("Starting LLM-assisted MongoDB search")
    
    # Process indications
    print("Processing indications text...")
    indications_text = indications
    if remove_acronyms:
        indications_text = rem_acronyms(indications_text)
    # Escape quotes before sending to LLM
    escaped_indications_text = escape_quotes(indications_text)
    indications_keywords = get_keywords_synonyms(escaped_indications_text, num_keywords, num_synonyms)
    print(f"Extracted {len(indications_keywords)} keywords from indications")
    
    # Process device details
    print("Processing device details text...")
    device_details_text = device_details
    if remove_acronyms:
        device_details_text = rem_acronyms(device_details_text)
    # Escape quotes before sending to LLM
    escaped_device_details_text = escape_quotes(device_details_text)
    device_details_keywords = get_keywords_synonyms(escaped_device_details_text, num_keywords, num_synonyms)
    print(f"Extracted {len(device_details_keywords)} keywords from device details")
    
    # Process operating principle
    print("Processing operating principle text...")
    operating_principle_text = operating_principle
    if remove_acronyms:
        operating_principle_text = rem_acronyms(operating_principle_text)
    # Escape quotes before sending to LLM
    escaped_operating_principle_text = escape_quotes(operating_principle_text)
    operating_principle_keywords = get_keywords_synonyms(escaped_operating_principle_text, num_keywords, num_synonyms)
    print(f"Extracted {len(operating_principle_keywords)} keywords from operating principle")
    
    # Perform vector searches using original text (not escaped)
    print("Performing vector search on indications...")
    indications_results = vector_search(
        collection, 
        indications_text,  # Using the original text
        indications_keywords,
        knumber_field_name,
        text_field_name,
        text_field_embed_name,
        index_field_name_1,
        model_id, 
        dimensions, 
        normalize,
        num_of_NN,
        num_of_results_to_return,
        is_ENN
    )
    
    print("Performing vector search on device details...")
    device_details_results = vector_search(
        collection, 
        device_details_text,  # Using the original text
        device_details_keywords,
        knumber_field_name,
        text_field_name,
        text_field_embed_name,
        index_field_name_1,
        model_id, 
        dimensions, 
        normalize,
        num_of_NN,
        num_of_results_to_return,
        is_ENN
    )
    
    print("Performing vector search on operating principle...")
    operating_principle_results = vector_search(
        collection, 
        operating_principle_text,  # Using the original text
        operating_principle_keywords,
        knumber_field_name,
        text_field_name,
        text_field_embed_name,
        index_field_name_1,
        model_id, 
        dimensions, 
        normalize,
        num_of_NN,
        num_of_results_to_return,
        is_ENN
    )
    
    # Convert results to DataFrames
    print("Processing results...")
    indications_df = pd.DataFrame(indications_results)
    device_details_df = pd.DataFrame(device_details_results)
    operating_principle_df = pd.DataFrame(operating_principle_results)
    
    # Group by K-number and get max score for each search
    indications_grouped = indications_df.groupby(knumber_field_name)['score'].max().reset_index()
    indications_grouped.rename(columns={knumber_field_name: 'knumber'}, inplace=True)
    
    device_details_grouped = device_details_df.groupby(knumber_field_name)['score'].max().reset_index()
    device_details_grouped.rename(columns={knumber_field_name: 'knumber'}, inplace=True)
    
    operating_principle_grouped = operating_principle_df.groupby(knumber_field_name)['score'].max().reset_index()
    operating_principle_grouped.rename(columns={knumber_field_name: 'knumber'}, inplace=True)
    
    # Combine results
    print("Combining search results...")
    combined_results = combine_3_search_results(
        indications_grouped,
        device_details_grouped,
        operating_principle_grouped,
        weights
    )
    
    print(f"Found {len(combined_results)} combined results")
    return combined_results 