from fastapi import APIRouter, HTTPException
from fastapi.responses import FileResponse
from starlette.background import BackgroundTask
import os

from app.utils.pdf import download_pdf_from_s3
from app.core.config import get_settings

settings = get_settings()

router = APIRouter(prefix="", tags=["pdf"])


@router.get("/getPDF/{knumber}")
async def get_kletter_pdf(knumber: str):
    """
    Return the PDF file for a given k-number.
    
    Args:
        knumber: The k-number of the PDF to retrieve
    
    Returns:
        FileResponse: The PDF file
    """
    try:
        # Ensure knumber has .pdf extension
        filename = knumber if knumber.endswith('.pdf') else knumber + '.pdf'
        
        # Download the file from S3
        local_file_path = download_pdf_from_s3(filename)
        
        if not local_file_path:
            raise HTTPException(status_code=404, detail="PDF file not found")
            
        def cleanup_file():
            try:
                if os.path.exists(local_file_path):
                    os.remove(local_file_path)
                    print(f"Successfully deleted {local_file_path}")
            except Exception as e:
                print(f"Error deleting file {local_file_path}: {str(e)}")
            
        # Return the file and clean up afterwards
        return FileResponse(
            path=local_file_path,
            filename=filename,
            media_type="application/pdf",
            background=BackgroundTask(cleanup_file)
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e)) 