from fastapi import FastAPI, Depends
from fastapi.middleware.cors import CORSMiddleware
import os
import asyncio
from concurrent.futures import ThreadPoolExecutor
import multiprocessing
from contextlib import asynccontextmanager

from app.services.mongodb import mongo_client, mongo_collection, initialize_db, close_db_connection, get_db
from app.routers import predicates, similarities, getpdf
from app.auth.aws.authorizer import cognito_jwt_authorizer_access_token as access_token
from app.core.config import get_settings

settings = get_settings()

# Set environment variables
# os.environ["TOKENIZERS_PARALLELISM"] = "false" # Removed as HF tokenizers library not directly used client-side

# Configure thread pool - REMOVED as it wasn't used and duplicated
# MAX_WORKERS = min(20, multiprocessing.cpu_count() * 2)
# thread_pool = ThreadPoolExecutor(max_workers=MAX_WORKERS)
# print("MAX_WORKERS: ", MAX_WORKERS)

# Define lifespan context manager for MongoDB connection
@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup code (runs before the app starts)
    settings = get_settings() # Get settings instance
    initialize_db(db_name=settings.MONGO_DB_NAME, collection_name=settings.MONGO_COLLECTION_NAME)
    # print("MongoDB connection established") # Message now printed within initialize_db
    
    yield  # The app runs here
    
    # Shutdown: close the connection
    close_db_connection()

# Create FastAPI app with lifespan
app = FastAPI(
    title="Medical Device Predicates API",
    description="API for finding and analyzing predicate devices for FDA 510(k) submissions",
    version="2.0.0",
    lifespan=lifespan,
    # Remove the following line if you want to test without having to deal with access tokens
    # though this scenario should only be for local testing.
    # dependencies=[Depends(access_token)]
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    #allow_origins=["https://*seai*"],
    allow_origins=["*"],  # Uncomment for local testing
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Override the dependency in routers
app.dependency_overrides[lambda: None] = get_db

# Include routers
app.include_router(predicates.router)
app.include_router(similarities.router)
app.include_router(getpdf.router)

# Root endpoint
@app.get("/", tags=["root"])
async def root():
    return {"message": "Welcome to the Medical Device Predicates API"}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app.main:app", host="0.0.0.0", port=8000, reload=True) 