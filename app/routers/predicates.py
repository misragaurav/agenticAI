from fastapi import APIRouter, HTTPException, Request, Depends
import pandas as pd
import time
import json
import asyncio
from concurrent.futures import ThreadPoolExecutor
import multiprocessing
import datetime
from typing import List, Tuple, Dict, Any

from app.models.schemas import (
    PredicateRequest, 
    PredicateResponse, 
    DetailDict, 
    DeviceResult, 
    DeviceMetadata,
    IndicsDevDetailsOpPrincipal,
    PredicateKnumberResponse,
    PredicateDetailsRequest,
    PredicateDetailsResponse,
    PredicateDetails,
    KeywordSynonyms,
    TextAnalysis,
    EnhancedPredicateResponse
)
from app.services.mongodb import get_mongo_fields
from app.services.search import query_mongo_db, query_mongo_LLM_assist, enrich_query_with_keywords
from app.services.llm import LLMClient, extract_paragraph, get_device_attributes, get_keywords_synonyms, reranker_LLM
from app.core.config import get_settings, KEYWORD_EXTRACTION_COUNT, SYNONYM_COUNT_PER_KEYWORD, SEARCH_WEIGHTS, SEARCH_WEIGHTS_TUPLE
from app.services.mongodb import get_db
from app.utils.getknumbertextfromDB import escape_quotes

settings = get_settings()

# Create router
router = APIRouter(prefix="", tags=["predicates"])

# Initialize thread pool - REMOVED as it wasn't used and duplicated
# MAX_WORKERS = min(20, multiprocessing.cpu_count() * 2)
# thread_pool = ThreadPoolExecutor(max_workers=MAX_WORKERS)

# Initialize LLM client
llm_client = LLMClient(settings.LLM_MODEL)
client_LLM = llm_client.client
model_LLM = llm_client.get_model_name()

# Helper function to process a single K-number result
def _process_kletter_result(kletter_info, mongo_collection) -> DeviceResult:
    """
    Processes a dictionary containing K-number and score, fetches details from MongoDB,
    and returns a formatted DeviceResult object.
    (This is synchronous because get_mongo_fields uses synchronous PyMongo).
    """
    knumber = kletter_info['knumber']
    score = round(kletter_info['score'], 3)

    # Get MongoDB fields
    mongo_fields = get_mongo_fields(knumber, mongo_collection)

    return DeviceResult(
        k_number=knumber,
        search_similarity_score=score,
        indications=mongo_fields["Indications for Use"],
        device_description=mongo_fields["Device Description"],
        operating_principle=mongo_fields["Operating Principle"],
        device_details=DetailDict(details={}),  # Placeholder
        differences="diffs",  # Placeholder
        similarities="sims",  # Placeholder
        metadata=DeviceMetadata(
            created_at="2024-01-01",  # Placeholder
            updated_at="2024-01-02",  # Placeholder
            version="1.0"  # Placeholder, consider setting appropriately
        ),
        device_name=mongo_fields["Device Name"],
        regulation_number=mongo_fields["Regulation Number"],
        regulation_name=mongo_fields["Regulation Name"],
        regulatory_class=mongo_fields["Regulatory Class"],
        product_code=mongo_fields["Product Code"],
        received_date=mongo_fields["Received"],
    )

# NEW synchronous helper for text processing & query enrichment
def _process_text_and_enrich_query(text: str, text_type: str) -> Tuple[List[KeywordSynonyms], str]:
    """
    Processes a single text field: escapes quotes, gets keywords/synonyms, 
    enriches the query, and returns analysis data + enriched query.
    (This is synchronous because get_keywords_synonyms is synchronous)
    """
    print(f"\nProcessing {text_type} text...")
    escaped_text = escape_quotes(text)
    # Assuming get_keywords_synonyms is synchronous (if it becomes async, this needs adjustment)
    keywords_dict = get_keywords_synonyms(escaped_text)
    # Enrich using the ORIGINAL text, not the escaped one
    enriched_query = enrich_query_with_keywords(text, keywords_dict)
    print(f"{text_type.capitalize()} query enriched: {enriched_query[:100]}...")
    
    keyword_synonyms_list = [
        KeywordSynonyms(keyword=k, synonyms=v) for k, v in keywords_dict.items()
    ]
    print(f"Extracted {len(keywords_dict)} keywords for {text_type}")
    
    return keyword_synonyms_list, enriched_query

# NEW synchronous helper to prepare data for the reranker LLM
def _prepare_reranker_input(results_list: List[DeviceResult], request_data: PredicateRequest) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    """
    Prepares the input data structures needed for the reranker_LLM function.

    Args:
        results_list: The list of DeviceResult objects from the initial search.
        request_data: The original PredicateRequest object containing user criteria.

    Returns:
        A tuple containing: 
        - reranker_input_results: List of dicts for candidate devices.
        - reranker_request_data: Dict representing the user's input criteria.
    """
    # Convert Pydantic models to simple dicts for the prompt
    reranker_input_results = [
        {
            "k_number": res.k_number,
            # Apply escape_quotes to each field
            "indications": escape_quotes(res.indications),
            "device_description": escape_quotes(res.device_description),
            "operating_principle": escape_quotes(res.operating_principle)
        } 
        for res in results_list # Use the list of DeviceResult objects
    ]
    # Convert input request Pydantic model to dict
    # Use model_dump() for Pydantic v2+
    reranker_request_data = request_data.model_dump()
    
    return reranker_input_results, reranker_request_data

# @router.post("/getPredicates", response_model=PredicateResponse)
# async def get_predicates(request: PredicateRequest, raw_request: Request, mongo_collection=Depends(get_db)):
#     try:
#         # Log the raw request for debugging
#         body = await raw_request.json()
#         print(f"Received request: {body}")
        
#         start_time = time.time()

#         # Validation checks
#         if request.number_of_predicates < 1:
#             raise HTTPException(
#                 status_code=422,
#                 detail="number_of_predicates must be greater than 0"
#             )

#         if not all([request.indications, request.device_details, request.operating_principle]):
#             raise HTTPException(
#                 status_code=422,
#                 detail="indications, device_details, and operating_principle cannot be empty"
#             )

#         # Use device_attributes directly as the details
#         detail_dict = DetailDict(details=request.device_attributes or {})

#         print("Progress: 10% - Querying MongoDB for similar documents...")
#         k_letters_n_scores = query_mongo_db(
#             mongo_collection, 
#             request.indications, 
#             model_id=settings.EMBEDDING_MODEL_ID, 
#             dimensions=settings.EMBEDDING_DIMENSIONS, 
#             normalize=True
#         )
        
#         print("Progress: 20% - Processing query results...")
#         df = pd.DataFrame(k_letters_n_scores)
#         df['metadata'] = df['metadata'].apply(json.loads)
#         df['knumber'] = df['metadata'].apply(lambda x: x['source'].split("/")[-1].split(".")[0])
#         df['filepath'] = df['metadata'].apply(lambda x: x['source'])

#         # max score instead of mean score
#         grouped_df = df.groupby('knumber')['score'].max().reset_index()
#         top_kletters = grouped_df.nlargest(request.number_of_predicates, 'score')
        
#         # Process all k-letters in parallel using run_in_executor
#         loop = asyncio.get_event_loop()
#         tasks = [
#             loop.run_in_executor(None, _process_kletter_result, row.to_dict(), mongo_collection) 
#             for _, row in top_kletters.iterrows()
#         ]
#         results_list = await asyncio.gather(*tasks)

#         print("Progress: 100% - Processing complete")
#         end_time = time.time()
#         execution_time = end_time - start_time
#         print(f"\nTotal execution time: {execution_time:.2f} seconds")

#         return PredicateResponse(details=detail_dict, results=results_list)

#     except Exception as e:
#         print(f"Unexpected error: {str(e)}")
#         raise HTTPException(status_code=500, detail=str(e))

@router.post("/getPredicatesAllFieldsKeywordsSynonyms", response_model=EnhancedPredicateResponse)
async def get_predicates_using_all_fields_and_keywords(
    request: PredicateRequest,
    raw_request: Request,
    mongo_collection=Depends(get_db),
    enrich_query: bool = False,
    perform_reranking: bool = True, # New flag for reranking
    indications_weight: float = SEARCH_WEIGHTS["indications"],
    device_details_weight: float = SEARCH_WEIGHTS["device_details"],
    operating_principle_weight: float = SEARCH_WEIGHTS["operating_principle"]
):
    try:
        # Log the raw request for debugging (no need to check for None now)
        try:
            body = await raw_request.json()
            print(f"Raw Received Request (/getPredicatesAll...): {body}")
        except Exception as log_e:
            print(f"Could not log raw request body: {log_e}")
            
        start_time = time.time()

        # Validation checks
        if request.number_of_predicates < 1:
            raise HTTPException(
                status_code=422,
                detail="number_of_predicates must be greater than 0"
            )

        if not all([request.indications, request.device_details, request.operating_principle]):
            raise HTTPException(
                status_code=422,
                detail="indications, device_details, and operating_principle cannot be empty"
            )
            
        # Create weights tuple with custom values
        custom_weights = (indications_weight, device_details_weight, operating_principle_weight)
        
        # Validate that weights sum to 1.0
        if abs(sum(custom_weights) - 1.0) > 0.001:  # Using small epsilon for float comparison
            raise HTTPException(
                status_code=422,
                detail=f"Search weights must sum to 1.0, got: {sum(custom_weights):.3f} from {custom_weights}"
            )

        # Use device_attributes directly as the details
        detail_dict = DetailDict(details=request.device_attributes or {})

        # Initialize variables
        text_analysis_list = []
        enriched_queries = {
            "indications": request.indications, # Default to original text
            "device_details": request.device_details,
            "operating_principle": request.operating_principle
        }

        if enrich_query:
            # =====================================================================
            # STEP 1: CONCURRENTLY ANALYZE TEXT, EXTRACT KEYWORDS/SYNONYMS, AND ENRICH QUERIES
            # =====================================================================
            print("\n" + "="*80)
            print("STEP 1: CONCURRENTLY PROCESSING TEXT FIELDS")
            print("="*80)

            loop = asyncio.get_event_loop()
            
            # Schedule the synchronous helper function calls in the executor
            indications_task = loop.run_in_executor(None, _process_text_and_enrich_query, request.indications, "indications")
            device_details_task = loop.run_in_executor(None, _process_text_and_enrich_query, request.device_details, "device_details")
            operating_principle_task = loop.run_in_executor(None, _process_text_and_enrich_query, request.operating_principle, "operating_principle")

            # Gather results concurrently
            print("Waiting for concurrent text processing to complete...")
            results = await asyncio.gather(
                indications_task,
                device_details_task,
                operating_principle_task
            )
            print("Concurrent text processing finished.")

            # Unpack results
            indications_kws_list, enriched_indications_query = results[0]
            device_details_kws_list, enriched_device_details_query = results[1]
            operating_principle_kws_list, enriched_operating_principle_query = results[2]

            # Reconstruct text_analysis_list and update enriched_queries dictionary
            text_analysis_list = [
                TextAnalysis(text_type="indications", keywords_synonyms=indications_kws_list),
                TextAnalysis(text_type="device_details", keywords_synonyms=device_details_kws_list),
                TextAnalysis(text_type="operating_principle", keywords_synonyms=operating_principle_kws_list)
            ]
            enriched_queries = {
                "indications": enriched_indications_query,
                "device_details": enriched_device_details_query,
                "operating_principle": enriched_operating_principle_query
            }
            print("\nText analysis and query enrichment complete.")
        else:
            # Skip enrichment, use original text
            print("\n" + "="*80)
            print("STEP 1: SKIPPING TEXT PROCESSING/ENRICHMENT (enrich_query=False)")
            print("="*80)
            # text_analysis_list remains empty
            # enriched_queries keeps the original text from initialization

        # =====================================================================
        # STEP 2: Perform MongoDB LLM-assisted search with (potentially) enriched queries
        # =====================================================================
        print("\n" + "="*80)
        print("STEP 2: SEARCHING FOR SIMILAR PREDICATE DEVICES")
        print("="*80)

        # Use search weights from parameters instead of centralized config
        print(f"\nUsing weighted scoring:")
        print(f"- Indications: {indications_weight*100:.0f}% of total score")
        print(f"- Device details: {device_details_weight*100:.0f}% of total score") 
        print(f"- Operating principle: {operating_principle_weight*100:.0f}% of total score")

        print("\nQuerying MongoDB using multi-query search...")
        # Await the now-async function
        k_letters_n_scores = await query_mongo_LLM_assist(
            collection=mongo_collection,
            # Pass the pre-enriched queries
            enriched_indications_query=enriched_queries["indications"],
            enriched_device_details_query=enriched_queries["device_details"],
            enriched_operating_principle_query=enriched_queries["operating_principle"],
            # Pass other necessary parameters
            model_id=settings.EMBEDDING_MODEL_ID,
            dimensions=settings.EMBEDDING_DIMENSIONS,
            normalize=True,
            # Pass custom weights
            weights=custom_weights
        )
        
        # =====================================================================
        # STEP 3: Retrieve full predicate details
        # =====================================================================
        print("\nProcessing search results and retrieving details...")
        top_kletters_info = []
        for knumber, score in k_letters_n_scores:
            top_kletters_info.append({
                'knumber': knumber,
                'score': score
            })
            
        print("\nRetrieving predicate device details concurrently...")
        loop = asyncio.get_event_loop() 
        tasks = [
            loop.run_in_executor(None, _process_kletter_result, kletter_info, mongo_collection) 
            for kletter_info in top_kletters_info
        ]
        initial_results_list = await asyncio.gather(*tasks) 
        print(f"\nRetrieved details for {len(initial_results_list)} initial candidates")

        # =====================================================================
        # STEP 4: Perform LLM Reranking (Conditional)
        # =====================================================================
        final_results_list = initial_results_list # Default if reranking is skipped
        
        if perform_reranking:
            print("\n" + "="*80)
            print("STEP 4: PERFORMING LLM RERANKING")
            print("="*80)
            if not initial_results_list:
                print("Reranking Step: No initial results to rerank.")
            
            MAX_RERANKER_CANDIDATES = 15 # Define how many top k-letters to rerank
            candidates_for_reranking = initial_results_list[:MAX_RERANKER_CANDIDATES]
            print(f"Considering top {len(candidates_for_reranking)} candidates for reranking.")

                
            if not candidates_for_reranking:
                print("Reranking Step: No initial candidates to rerank.")
                final_results_list = [] # Ensure list is empty
            else:
                # 4a. Prepare data for reranker using the LIMITED list
                print("Preparing data for reranker...")
                reranker_input_candidates, reranker_request_criteria = _prepare_reranker_input(
                    candidates_for_reranking, request 
                )
                
                # 4b. Call reranker function
                # Note: reranker_LLM also has an internal limit, but we pass the pre-limited list
                print("Calling reranker LLM...")
                knumbers_to_remove = await loop.run_in_executor(
                    None, 
                    reranker_LLM, 
                    reranker_input_candidates,   # Arg 1 for reranker_LLM
                    reranker_request_criteria    # Arg 2 for reranker_LLM
                )
                
                # 4c. Filter the LIMITED list based on reranker output
                if knumbers_to_remove:
                    # Filter the list that was actually sent for reranking
                    reranked_candidates = [
                        res for res in candidates_for_reranking 
                        if res.k_number not in knumbers_to_remove
                    ]
                    print(f"Reranked results: {len(reranked_candidates)} candidates remaining from the top {len(candidates_for_reranking)} after removing {knumbers_to_remove}.")
                    final_results_list = reranked_candidates # Update the final list
                else:
                    final_results_list = candidates_for_reranking # Use the original top N if none removed
                    print(f"Reranker did not identify any candidates to remove from the top {len(candidates_for_reranking)}.")
        else:
             print("\n" + "="*80)
             print("STEP 4: SKIPPING LLM RERANKING (perform_reranking=False)")
             print("="*80)

        # Apply final limit (e.g., number_of_predicates requested) AFTER reranking
        final_limited_results = final_results_list[:request.number_of_predicates]
        print(f"Final result count (after limit): {len(final_limited_results)}")

        end_time = time.time()
        execution_time = end_time - start_time
        print(f"\nTotal execution time for /getPredicatesAllFieldsKeywordsSynonyms: {execution_time:.2f} seconds")

        # Return the enhanced response with text analysis and final results
        return EnhancedPredicateResponse(
            details=detail_dict, 
            text_analysis=text_analysis_list,
            results=final_limited_results # Return the potentially reranked & limited list
        )

    except Exception as e:
        print(f"Unexpected error in /getPredicatesAll...: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/getPredicateKnumbers", response_model=PredicateKnumberResponse)
async def get_predicate_knumbers(
    raw_request: Request,
    request: IndicsDevDetailsOpPrincipal,
    mongo_collection=Depends(get_db)
):
    """
    Return a list of predicate K-numbers and their similarity scores using the 
    advanced LLM-assisted search logic by calling get_predicates_using_all_fields_and_keywords.
    
    Args:
        request (IndicsDevDetailsOpPrincipal): Contains indications, device_details, and operating_principle
    
    Returns:
        PredicateKnumberResponse: Object containing a list of (knumber, score) tuples
    """
    try:
        start_time = time.time()
        loop = asyncio.get_event_loop() # Get loop for potential executor use
        
        # Log raw request at the start
        try:
            body = await raw_request.json()
            print(f"Raw Received Request (/getPredicateKnumbers): {body}")
        except Exception as log_e:
            # Log if reading/parsing fails, but don't stop execution
            print(f"Could not log raw request body in /getPredicateKnumbers: {log_e}")

        print("Calling advanced predicate search (without enrichment or reranking) to get K-numbers...")
        
        # 1. Create a PredicateRequest object from the input `request` (IndicsDevDetailsOpPrincipal)
        predicate_request_body = PredicateRequest(
            indications=request.indications,
            device_details=request.device_details,
            operating_principle=request.operating_principle,
            number_of_predicates=20, 
            device_attributes={}
        )
        
        # 2. Call the advanced search function, passing both request objects and disabling enrichment
        advanced_response = await get_predicates_using_all_fields_and_keywords(
            request=predicate_request_body, 
            raw_request=raw_request,                
            mongo_collection=mongo_collection,
            enrich_query=False, 
            perform_reranking=True # reranking
        )
        
        # 3. Extract K-number and score from the results 
        # The results here are already processed DeviceResult objects
        final_results = [
            (res.k_number, res.search_similarity_score) 
            for res in advanced_response.results
        ]

        # --- Reranking Step REMOVED from here --- 
        
        # Ensure sorting by score (descending)
        final_results.sort(key=lambda item: item[1], reverse=True)
        print(f"Advanced search returned {len(final_results)} K-numbers/scores.")
        
        # Final timing and return
        end_time = time.time()
        execution_time = end_time - start_time
        print(f"\nTotal execution time for /getPredicateKnumbers (incl. reranker): {execution_time:.2f} seconds")
        
        # Return the list
        return PredicateKnumberResponse(results=final_results)
        
    except Exception as e:
        print(f"Unexpected error in /getPredicateKnumbers: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/getPredicateDetails", response_model=PredicateDetailsResponse)
async def get_predicate_details(
    request: PredicateDetailsRequest, 
    mongo_collection=Depends(get_db)
):
    """
    Get detailed information about specific predicate devices.
    
    Args:
        request (PredicateDetailsRequest): List of knumbers to get details for
    
    Returns:
        PredicateDetailsResponse: Contains details for each requested predicate device
    """
    # Input validation
    if not request.knumbers:
        raise HTTPException(
            status_code=422,
            detail="List of knumbers cannot be empty"
        )

    start_time = time.time()

    async def process_kletter(knumber: str):
        try:
            # Get MongoDB fields
            try:
                mongo_fields = get_mongo_fields(knumber, mongo_collection)
            except Exception as e:
                print(f"Error fetching MongoDB fields for {knumber}: {str(e)}")
                mongo_fields = {
                    "Device Name": "Error fetching data from MongoDB",
                    "Regulation Number": "Error fetching data from MongoDB",
                    "Regulation Name": "Error fetching data from MongoDB",
                    "Regulatory Class": "Error fetching data from MongoDB",
                    "Product Code": "Error fetching data from MongoDB",
                    "Received": "Error fetching data from MongoDB",
                    "Device Description": "Error fetching data from MongoDB",
                    "Operating Principle": "Error fetching data from MongoDB",
                    "Indications for Use": "Error fetching data from MongoDB"
                }

            return PredicateDetails(
                k_number=knumber,
                device_description=mongo_fields["Device Description"],
                operating_principle=mongo_fields["Operating Principle"],
                indications=mongo_fields["Indications for Use"],
                device_name=mongo_fields["Device Name"],
                regulation_number=mongo_fields["Regulation Number"],
                regulation_name=mongo_fields["Regulation Name"],
                regulatory_class=mongo_fields["Regulatory Class"],
                product_code=mongo_fields["Product Code"],
                received_date=mongo_fields["Received"],
                submitter="Unknown",
                manufacturer="Unknown"
            )

        except Exception as e:
            print(f"Critical error processing {knumber}: {str(e)}")
            return PredicateDetails(
                k_number=knumber,
                device_description="Error processing document from MongoDB",
                operating_principle="Error processing document from MongoDB",
                indications="Error processing document from MongoDB",
                device_name="Error processing document from MongoDB",
                regulation_number="Error processing document from MongoDB",
                regulation_name="Error processing document from MongoDB",
                regulatory_class="Error processing document from MongoDB",
                product_code="Error processing document from MongoDB",
                received_date="Error processing document from MongoDB",
                submitter="Error processing document from MongoDB",
                manufacturer="Error processing document from MongoDB"
            )

    try:
        # Process all k-letters in parallel
        tasks = [process_kletter(knumber) for knumber in request.knumbers]
        results_list = await asyncio.gather(*tasks)
        
        end_time = time.time()
        print(f"Total execution time: {end_time - start_time:.2f} seconds")

        return PredicateDetailsResponse(results=results_list)

    except Exception as e:
        print(f"Error in parallel processing K numbers: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error processing predicates: {str(e)}")
    