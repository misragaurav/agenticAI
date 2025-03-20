from fastapi import APIRouter, HTTPException, Request, Depends
import pandas as pd
import time
import json
import asyncio
from concurrent.futures import ThreadPoolExecutor
import multiprocessing
import datetime

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
from app.services.search import query_mongo_db, query_mongo_LLM_assist, get_keywords_synonyms
from app.services.llm import LLMClient, extract_paragraph, get_device_attributes
from app.core.config import get_settings, KEYWORD_EXTRACTION_COUNT, SYNONYM_COUNT_PER_KEYWORD, SEARCH_WEIGHTS, SEARCH_WEIGHTS_TUPLE
from app.services.mongodb import get_db
from app.utils.getknumbertextfromDB import escape_quotes

settings = get_settings()

# Create router
router = APIRouter(prefix="", tags=["predicates"])

# Initialize thread pool
MAX_WORKERS = min(20, multiprocessing.cpu_count() * 2)
thread_pool = ThreadPoolExecutor(max_workers=MAX_WORKERS)

# Initialize LLM client
llm_client = LLMClient(settings.LLM_MODEL)
client_LLM = llm_client.client
model_LLM = llm_client.get_model_name()

@router.post("/getPredicates", response_model=PredicateResponse)
async def get_predicates(request: PredicateRequest, raw_request: Request, mongo_collection=Depends(get_db)):
    try:
        # Log the raw request for debugging
        body = await raw_request.json()
        print(f"Received request: {body}")
        
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

        # Use device_attributes directly as the details
        detail_dict = DetailDict(details=request.device_attributes or {})

        print("Progress: 10% - Querying MongoDB for similar documents...")
        k_letters_n_scores = query_mongo_db(
            mongo_collection, 
            request.indications, 
            model_id=settings.BEDROCK_MODEL_ID, 
            dimensions=settings.EMBEDDING_DIMENSIONS, 
            normalize=True
        )
        
        print("Progress: 20% - Processing query results...")
        df = pd.DataFrame(k_letters_n_scores)
        df['metadata'] = df['metadata'].apply(json.loads)
        df['knumber'] = df['metadata'].apply(lambda x: x['source'].split("/")[-1].split(".")[0])
        df['filepath'] = df['metadata'].apply(lambda x: x['source'])

        # max score instead of mean score
        grouped_df = df.groupby('knumber')['score'].max().reset_index()
        top_kletters = grouped_df.nlargest(request.number_of_predicates, 'score')
        

        async def process_kletter(row):
            knumber = row['knumber']
            score = round(row['score'], 3)
            
            # Get MongoDB fields - either:
            # 1. Call it directly since it's now synchronous
            mongo_fields = get_mongo_fields(knumber, mongo_collection)
            
            # OR 2. Keep the executor but don't await the function
            # mongo_fields = await asyncio.get_event_loop().run_in_executor(
            #     thread_pool,
            #     get_mongo_fields,
            #     knumber,
            #     mongo_collection
            # )

            return DeviceResult(
                k_number=knumber,
                search_similarity_score=score,
                indications=mongo_fields["Indications for Use"],
                device_description=mongo_fields["Device Description"],
                operating_principle=mongo_fields["Operating Principle"],
                device_details=DetailDict(details={}),
                differences="diffs",
                similarities="sims",
                metadata=DeviceMetadata(
                    created_at="2024-01-01",
                    updated_at="2024-01-02",
                    version="1.0"
                ),
                device_name=mongo_fields["Device Name"],
                regulation_number=mongo_fields["Regulation Number"],
                regulation_name=mongo_fields["Regulation Name"],
                regulatory_class=mongo_fields["Regulatory Class"],
                product_code=mongo_fields["Product Code"],
                received_date=mongo_fields["Received"],
            )

        # Process all k-letters in parallel
        tasks = [process_kletter(row) for _, row in top_kletters.iterrows()]
        results_list = await asyncio.gather(*tasks)

        print("Progress: 100% - Processing complete")
        end_time = time.time()
        execution_time = end_time - start_time
        print(f"\nTotal execution time: {execution_time:.2f} seconds")

        return PredicateResponse(details=detail_dict, results=results_list)

    except Exception as e:
        print(f"Unexpected error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/getPredicatesAllFieldsKeywordsSynonyms", response_model=EnhancedPredicateResponse)
async def get_predicates_using_all_fields_and_keywords(request: PredicateRequest, raw_request: Request, mongo_collection=Depends(get_db)):
    try:
        # Log the raw request for debugging
        body = await raw_request.json()
        print(f"Received request: {body}")
        
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

        # Use device_attributes directly as the details
        detail_dict = DetailDict(details=request.device_attributes or {})

        # =====================================================================
        # FIRST PART: Extract keywords and synonyms for text analysis
        # =====================================================================
        print("\n" + "="*80)
        print("STEP 1: ANALYZING TEXT AND EXTRACTING KEYWORDS/SYNONYMS")
        print("="*80)
        
        text_analysis_list = []
        
        # Process indications text
        print("\nProcessing indications text...")
        indications_text = request.indications
        escaped_indications_text = escape_quotes(indications_text)
        indications_keywords = get_keywords_synonyms(escaped_indications_text)
        
        # Add to text analysis
        indications_analysis = TextAnalysis(
            text_type="indications",
            keywords_synonyms=[
                KeywordSynonyms(keyword=k, synonyms=v) for k, v in indications_keywords.items()
            ]
        )
        text_analysis_list.append(indications_analysis)
        
        # Process device details text
        print("\nProcessing device details text...")
        device_details_text = request.device_details
        escaped_device_details_text = escape_quotes(device_details_text)
        device_details_keywords = get_keywords_synonyms(escaped_device_details_text)
        
        # Add to text analysis
        device_details_analysis = TextAnalysis(
            text_type="device_details",
            keywords_synonyms=[
                KeywordSynonyms(keyword=k, synonyms=v) for k, v in device_details_keywords.items()
            ]
        )
        text_analysis_list.append(device_details_analysis)
        
        # Process operating principle text
        print("\nProcessing operating principle text...")
        operating_principle_text = request.operating_principle
        escaped_operating_principle_text = escape_quotes(operating_principle_text)
        operating_principle_keywords = get_keywords_synonyms(escaped_operating_principle_text)
        
        # Add to text analysis
        operating_principle_analysis = TextAnalysis(
            text_type="operating_principle",
            keywords_synonyms=[
                KeywordSynonyms(keyword=k, synonyms=v) for k, v in operating_principle_keywords.items()
            ]
        )
        text_analysis_list.append(operating_principle_analysis)
        
        print("\nText analysis complete. Found:")
        print(f"- Indications: {len(indications_keywords)} keywords (of {KEYWORD_EXTRACTION_COUNT} requested)")
        print(f"- Device details: {len(device_details_keywords)} keywords (of {KEYWORD_EXTRACTION_COUNT} requested)")
        print(f"- Operating principle: {len(operating_principle_keywords)} keywords (of {KEYWORD_EXTRACTION_COUNT} requested)")
        print(f"- Each keyword has up to {SYNONYM_COUNT_PER_KEYWORD} synonyms")
        
        # =====================================================================
        # SECOND PART: Perform MongoDB LLM-assisted search
        # =====================================================================
        print("\n" + "="*80)
        print("STEP 2: SEARCHING FOR SIMILAR PREDICATE DEVICES")
        print("="*80)

        # Use search weights from centralized config
        print(f"\nUsing weighted scoring:")
        print(f"- Indications: {SEARCH_WEIGHTS['indications']*100:.0f}% of total score")
        print(f"- Device details: {SEARCH_WEIGHTS['device_details']*100:.0f}% of total score") 
        print(f"- Operating principle: {SEARCH_WEIGHTS['operating_principle']*100:.0f}% of total score")

        print("\nQuerying MongoDB using LLM-assisted search...")
        k_letters_n_scores = query_mongo_LLM_assist(
            collection=mongo_collection, 
            indications=request.indications,
            device_details=request.device_details,
            operating_principle=request.operating_principle,
            model_id=settings.BEDROCK_MODEL_ID, 
            dimensions=settings.EMBEDDING_DIMENSIONS, 
            normalize=True,
            remove_acronyms=False
            # weights parameter not specified, will use default from config
        )
        
        print("\nProcessing search results...")
        # Create a list of k-numbers and scores to process
        top_kletters = []
        for knumber, score in k_letters_n_scores[:request.number_of_predicates]:
            top_kletters.append({
                'knumber': knumber,
                'score': score
            })

        async def process_kletter(kletter_info):
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
                device_details=DetailDict(details={}),
                differences="diffs",
                similarities="sims",
                metadata=DeviceMetadata(
                    created_at="2024-01-01",
                    updated_at="2024-01-02",
                    version="1.0"
                ),
                device_name=mongo_fields["Device Name"],
                regulation_number=mongo_fields["Regulation Number"],
                regulation_name=mongo_fields["Regulation Name"],
                regulatory_class=mongo_fields["Regulatory Class"],
                product_code=mongo_fields["Product Code"],
                received_date=mongo_fields["Received"],
            )

        # Process all k-letters in parallel
        print("\nRetrieving predicate device details...")
        tasks = [process_kletter(kletter_info) for kletter_info in top_kletters]
        results_list = await asyncio.gather(*tasks)

        print("\nFound {} predicate devices".format(len(results_list)))
        end_time = time.time()
        execution_time = end_time - start_time
        print(f"\nTotal execution time: {execution_time:.2f} seconds")

        # Return the enhanced response with text analysis first, then results
        return EnhancedPredicateResponse(
            details=detail_dict, 
            text_analysis=text_analysis_list,
            results=results_list
        )

    except Exception as e:
        print(f"Unexpected error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/getPredicateKnumbers", response_model=PredicateKnumberResponse)
async def get_predicate_knumbers(
    request: IndicsDevDetailsOpPrincipal, 
    mongo_collection=Depends(get_db)
):
    """
    Return a list of predicate K-numbers and their similarity scores based on the input device details.
    
    Args:
        request (IndicsDevDetailsOpPrincipal): Contains indications, device_details, and operating_principle
    
    Returns:
        PredicateKnumberResponse: Object containing a list of (knumber, score) tuples
    """
    try:
        start_time = time.time()
        
        # Validate inputs
        if not all([request.indications, request.device_details, request.operating_principle]):
            raise HTTPException(
                status_code=422,
                detail="indications, device_details, and operating_principle cannot be empty"
            )
        
        print("Progress: 10% - Querying MongoDB for similar documents...")
        k_letters_n_scores = query_mongo_db(
            mongo_collection, 
            request.indications, 
            model_id=settings.BEDROCK_MODEL_ID, 
            dimensions=settings.EMBEDDING_DIMENSIONS, 
            normalize=True
        )
        
        print("Progress: 20% - Processing query results...")
        df = pd.DataFrame(k_letters_n_scores)
        df['metadata'] = df['metadata'].apply(json.loads)
        df['knumber'] = df['metadata'].apply(lambda x: x['source'].split("/")[-1].split(".")[0])
        df['filepath'] = df['metadata'].apply(lambda x: x['source'])

        # max score instead of mean score
        grouped_df = df.groupby('knumber')['score'].max().reset_index()
        
        # Sort by score in descending order
        sorted_df = grouped_df.sort_values('score', ascending=False)
        
        # Convert to list of tuples
        results = [(row['knumber'], float(round(row['score'], 3))) for _, row in sorted_df.iterrows()]
        
        print("Progress: 100% - Processing complete")
        end_time = time.time()
        execution_time = end_time - start_time
        print(f"\nTotal execution time: {execution_time:.2f} seconds")
        
        return PredicateKnumberResponse(results=results)
        
    except Exception as e:
        print(f"Unexpected error: {str(e)}")
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
    