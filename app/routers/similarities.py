from fastapi import APIRouter, HTTPException, Depends
import asyncio
from starlette.background import BackgroundTask

from app.models.schemas import SimDiffRequest, SimDiffResponse, DetailedSimDiffRequest
from app.utils.pdf import extract_text_from_pdf, extract_manuals_text
from app.services.llm import LLMClient, write_sim_diff_discussion, write_detailed_sim_diff_discussion
from app.core.config import get_settings
from app.utils.getknumbertextfromDB import get_text_from_db
from app.services.mongodb import get_db

settings = get_settings()

router = APIRouter(prefix="", tags=["similarities"])

# Initialize LLM client
llm_client = LLMClient(settings.LLM_MODEL)
client_LLM = llm_client.client
model_LLM = llm_client.get_model_name()


@router.post("/getSimilarityDifferences", response_model=SimDiffResponse)
async def get_similarity_differences(request: SimDiffRequest, mongo_collection=Depends(get_db)):
    """
    Generate similarities and differences between a user's device and a predicate device.
    
    Args:
        request (SimDiffRequest): Contains knumber, indications, device_details, operating_principle
    
    Returns:
        SimDiffResponse: Contains similarities and differences paragraphs
    """
    try:
        # Get text from database instead of extracting from PDF
        print(f"Fetching text from database for K-number {request.knumber}...")
        predicate_text = get_text_from_db(
            request.knumber,
            mongo_collection
        )

        # Combine input texts
        given_device_text = (
            request.indications + "\n" + 
            request.device_details + "\n" + 
            request.operating_principle
        )

        # Generate similarities and differences with enhanced parameters
        similarities = write_sim_diff_discussion(
            given_device_text, 
            predicate_text, 
            "similarities", 
            llm_client,
            model_LLM
        )
        differences = write_sim_diff_discussion(
            given_device_text, 
            predicate_text, 
            "differences", 
            llm_client,
            model_LLM
        )

        return SimDiffResponse(
            similarities=similarities,
            differences=differences
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/getDetailedSimDiffs", response_model=SimDiffResponse)
async def get_detailed_sim_diffs(request: DetailedSimDiffRequest):
    """
    Generate detailed similarities and differences between user's device and predicate device
    using the device manuals and k-letter.
    
    Args:
        request (DetailedSimDiffRequest): Contains knumber, predicate_manuals_s3_path, and user_device_manuals_s3_path
    
    Returns:
        SimDiffResponse: Object containing detailed similarities and differences paragraphs
    """
    # Using a bigger model for this task
    detailed_llm_client = LLMClient("deepseek-r1-distill-llama-70b")
    #client_detailed_LLM = detailed_llm_client.client
    model_detailed_LLM = detailed_llm_client.get_model_name()

    try:
        # Ensure knumber has .pdf extension
        filename = request.knumber if request.knumber.endswith('.pdf') else request.knumber + '.pdf'
        
        # Extract text from the k-letter PDF
        print(f"Extracting text from k-letter PDF {filename}...")
        kletter_text = extract_text_from_pdf(settings.S3_BUCKET_NAME, settings.S3_BUCKET_PATH, filename, 4)
        
        # Extract text from predicate device manuals
        print(f"Extracting text from predicate device manuals...")
        predicate_manuals_text = extract_manuals_text(request.predicate_manuals_s3_path)
        
        # Extract text from user's device manuals
        print(f"Extracting text from user's device manuals...")
        user_manuals_text = extract_manuals_text(request.user_device_manuals_s3_path)
        
        # Generate detailed similarities and differences
        print("Generating detailed similarities and differences...")
        similarities, differences = write_detailed_sim_diff_discussion(
            kletter_text,
            predicate_manuals_text,
            user_manuals_text,
            detailed_llm_client,
            model_detailed_LLM
        )
        
        return SimDiffResponse(
            similarities=similarities,
            differences=differences
        )
        
    except Exception as e:
        print(f"Error in get_detailed_sim_diffs: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e)) 
    


# @router.post("/getSimilarityDifferencesFromPDF", response_model=SimDiffResponse)
# async def get_similarity_differences_from_pdf(request: SimDiffRequest):
#     """
#     Generate similarities and differences between a user's device and a predicate device.
    
#     Args:
#         request (SimDiffRequest): Contains knumber, indications, device_details, operating_principle
    
#     Returns:
#         SimDiffResponse: Contains similarities and differences paragraphs
#     """
#     try:
#         # Ensure knumber has .pdf extension
#         filename = request.knumber if request.knumber.endswith('.pdf') else request.knumber + '.pdf'

#         # Extract text from PDF
#         print(f"Extracting text from PDF {filename}...")
#         predicate_text = extract_text_from_pdf(
#             settings.S3_BUCKET_NAME, 
#             settings.S3_BUCKET_PATH, 
#             filename, 
#             4
#         )

#         # Combine input texts
#         given_device_text = (
#             request.indications + "\n" + 
#             request.device_details + "\n" + 
#             request.operating_principle
#         )

#         # Generate similarities and differences
#         similarities = write_sim_diff_discussion(
#             given_device_text, 
#             predicate_text, 
#             "similarities", 
#             llm_client,
#             model_LLM
#         )
#         differences = write_sim_diff_discussion(
#             given_device_text, 
#             predicate_text, 
#             "differences", 
#             llm_client,
#             model_LLM
#         )

#         return SimDiffResponse(
#             similarities=similarities,
#             differences=differences
#         )

#     except Exception as e:
#         raise HTTPException(status_code=500, detail=str(e))