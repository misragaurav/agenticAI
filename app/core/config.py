import boto3
from botocore.exceptions import ClientError
import json
from functools import lru_cache
from pydantic_settings import BaseSettings, SettingsConfigDict

# Keyword extraction settings
KEYWORD_EXTRACTION_COUNT = 4  # Number of keywords to extract
SYNONYM_COUNT_PER_KEYWORD = 2  # Number of synonyms per keyword

# Search component weights (must sum to 1.0)
SEARCH_WEIGHTS = {
    "indications": 0.7,       # 60% weight for indications matches
    "device_details": 0.15,    # 20% weight for device details matches
    "operating_principle": 0.15 # 20% weight for operating principle matches
}

# Tuple form of weights for direct use in functions
SEARCH_WEIGHTS_TUPLE = (
    SEARCH_WEIGHTS["indications"],
    SEARCH_WEIGHTS["device_details"],
    SEARCH_WEIGHTS["operating_principle"]
)

class AppSettings(BaseSettings):
    # AWS Cognito settings moved to app/auth/settings.py
    AWS_SECRETS_MANAGER_REGION: str = "us-west-2" 

    EMBEDDING_MODEL_ID: str = "amazon.titan-embed-text-v2:0"
    EMBEDDING_DIMENSIONS: int = 1024
    S3_BUCKET_NAME: str = "klettersfast"
    S3_BUCKET_PATH: str = "klettersfast/"
    LLM_MODEL: str = "deepseek-r1-distill-llama-70b"
    
    # MongoDB Settings
    MONGO_DB_NAME: str = "se"
    MONGO_COLLECTION_NAME: str = "fixed_150tok_1024dims"

@lru_cache
def get_settings():
    return AppSettings()


def get_secret():
    """
    Get secrets from AWS Secrets Manager.
    
    Returns:
        dict: Dictionary containing secrets
    """
    secret_name = "openai_hf_mongo_keys"
    # Use the region defined in settings
    settings = get_settings()
    region_name = settings.AWS_SECRETS_MANAGER_REGION

    session = boto3.session.Session()
    client = session.client(
        service_name='secretsmanager',
        region_name=region_name
    )

    try:
        get_secret_value_response = client.get_secret_value(
            SecretId=secret_name
        )
        # Parse the JSON string into a dictionary
        secrets = json.loads(get_secret_value_response['SecretString'])
        return secrets
    except ClientError as e:
        # For a list of exceptions thrown, see
        # https://docs.aws.amazon.com/secretsmanager/latest/apireference/API_GetSecretValue.html
        raise e 