import boto3
import json
from app.core.config import get_settings

settings = get_settings()

def get_embedding_from_bedrock(input_text, model_id=None, dimensions=None, normalize=True):
    """
    Get embeddings from AWS Bedrock using the Titan embedding model.
    
    Args:
        input_text (str): The text to embed
        model_id (str): Model ID to use
        dimensions (int): Embedding dimensions (256, 512, or 1024)
        normalize (bool): Whether to normalize the output embedding
    
    Returns:
        dict: Contains 'embedding' (list of floats), 'input_token_count' (int)
    """
    # Use settings if parameters not provided
    model_id = model_id or settings.BEDROCK_MODEL_ID
    dimensions = dimensions or settings.EMBEDDING_DIMENSIONS
    
    # Initialize the Bedrock Runtime client
    bedrock_runtime = boto3.client('bedrock-runtime', region_name='us-west-2')
    
    # Create the request payload
    payload = {
        "inputText": input_text,
        "dimensions": dimensions,
        "normalize": normalize
    }

    # Invoke the model
    response = bedrock_runtime.invoke_model(
        body=json.dumps(payload),
        modelId=model_id,
        accept="application/json",
        contentType="application/json"
    )

    # Parse the response
    response_body = json.loads(response['body'].read())
    
    return response_body.get('embedding') 