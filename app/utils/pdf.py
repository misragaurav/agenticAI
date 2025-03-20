import fitz  # PyMuPDF's actual import name
from typing import Optional, List
import boto3
import os
import uuid
from app.core.config import get_settings

settings = get_settings()

def download_pdf_from_s3(filename: str, s3_bucket_name=None, s3_bucket_path=None) -> Optional[str]:
    """
    Download a PDF file from S3 to local storage.
    
    Args:
        filename: Name of the PDF file
        s3_bucket_name: S3 bucket name (defaults to settings)
        s3_bucket_path: S3 bucket path (defaults to settings)
    
    Returns:
        Optional[str]: Local file path if successful, None otherwise
    """
    s3_bucket_name = s3_bucket_name or settings.S3_BUCKET_NAME
    s3_bucket_path = s3_bucket_path or settings.S3_BUCKET_PATH
    
    # Create unique filename to avoid conflicts
    unique_id = str(uuid.uuid4())[:8]
    local_file_path = f"./tmp_{unique_id}_{filename}"
    
    print(f"Downloading PDF file '{filename}' from S3...")
    # Initialize S3 client
    s3 = boto3.client('s3', region_name='us-west-2')
    s3_path = s3_bucket_name+'/'+s3_bucket_path+f"{filename}"

    try:
        s3.download_file(s3_bucket_name, s3_bucket_path+filename, local_file_path)
        print("Download completed successfully")
    except Exception as e:
        print(f"Error downloading {s3_path} from S3: {str(e)}")
        return None
    return local_file_path


def extract_text_from_pdf(s3_bucket_name: str, s3_bucket_path: str, filename: str, page_to_start_extraction_from: int) -> str:
    """
    Extract text from a PDF file.
    
    Args:
        s3_bucket_name: S3 bucket name
        s3_bucket_path: S3 bucket path
        filename: Name of the PDF file
        page_to_start_extraction_from: First page to extract (1-indexed)
    
    Returns:
        str: Extracted text or None if failed
    """
    s3_path = s3_bucket_name+'/'+s3_bucket_path+f"{filename}"

    local_file_path = download_pdf_from_s3(filename, s3_bucket_name, s3_bucket_path)
    if not local_file_path:
        return None

    print(f"Extracting text from PDF starting from page {page_to_start_extraction_from}...")
    text = ""
    try:
        # Open and extract text from PDF
        with fitz.open(local_file_path) as doc:
            print(f"PDF opened successfully. Total pages: {len(doc)}")
            # Join all pages' text with newlines
            text = "\n".join([page.get_text() for page_num, page in enumerate(doc) if page_num >= page_to_start_extraction_from-1])
            if not text:
                raise ValueError(f"No text found in PDF at {s3_path}")
            print("Text extraction completed successfully")
            
    except fitz.FileDataError as e:
        print(f"Error: Could not open PDF file at {s3_path}: {str(e)}")
        return None

    except Exception as e:
        print(f"Unexpected error: {str(e)}")
        return None 

    def cleanup_file(local_file_path):
        try:
            if os.path.exists(local_file_path):
                os.remove(local_file_path)
                print(f"Successfully deleted {local_file_path}")
        except Exception as e:
            print(f"Error deleting file {local_file_path}: {str(e)}")

    cleanup_file(local_file_path)

    return text


async def extract_manuals_text(s3_path: str) -> str:
    """
    Extract text from all PDF manuals in the given S3 path.
    
    Args:
        s3_path: S3 path to the directory containing PDF manuals
    
    Returns:
        str: Combined text from all PDFs
    """
    try:
        # Initialize S3 client
        s3 = boto3.client('s3', region_name='us-west-2')
        
        # List all objects in the S3 path
        response = s3.list_objects_v2(Bucket=settings.S3_BUCKET_NAME, Prefix=s3_path)
        
        if 'Contents' not in response:
            return ""
        
        # Extract text from each PDF
        combined_text = ""
        for obj in response['Contents']:
            if obj['Key'].endswith('.pdf'):
                filename = obj['Key'].split('/')[-1]
                text = extract_text_from_pdf(settings.S3_BUCKET_NAME, s3_path, filename, 4)
                if text:
                    combined_text += "\n\n" + text
        
        return combined_text
        
    except Exception as e:
        print(f"Error extracting manuals text: {str(e)}")
        return "" 