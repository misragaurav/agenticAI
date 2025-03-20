from openai import OpenAI
import boto3
import os
from app.core.config import get_settings, get_secret

settings = get_settings()

def remove_think_tags(text: str) -> str:
    """
    Remove all text contained within <think> tags from the input text.
    
    Args:
        text (str): The input text containing possible <think> tags
        
    Returns:
        str: The text with all <think> sections removed
    """
    result = text
    while '<think>' in result and '</think>' in result:
        start = result.find('<think>')
        end = result.find('</think>') + len('</think>')
        result = result[:start].strip() + ' ' + result[end:].strip()
    return result


class LLMClient:
    """
    A class for interacting with various LLM providers.
    """
    
    def __init__(self, model_type=None):
        """
        Initialize the LLM client.
        
        Args:
            model_type (str): Type of model to use (defaults to settings)
        """
        self.model_type = model_type or settings.LLM_MODEL
        self.client = self._initialize_client()
    
    def _initialize_client(self):
        """
        Initialize the appropriate client based on model type.
        
        Returns:
            Client object for the specified model type
        """
        secrets = get_secret()
        
        # OpenAI models
        if "gpt" in self.model_type.lower():
            return OpenAI(api_key=secrets["openAI_API_key"])
        
        # Hugging Face models
        elif self.model_type == "llama":
            return OpenAI(
                base_url="https://api-inference.huggingface.co/v1/",
                api_key=secrets["hf_API_key"]
            )
        
        # Groq models
        elif any(name in self.model_type.lower() for name in ["deepseek", "qwen"]) or (
            "llama" in self.model_type.lower() and self.model_type != "llama"
        ):
            from groq import Groq
            return Groq(api_key=secrets["groq_API_key"])
        
        else:
            # Default to OpenAI
            return OpenAI(api_key=secrets["openAI_API_key"])
    
    def get_model_name(self):
        """
        Get the full model name for the selected model.
        
        Returns:
            str: Full model name
        """
        if self.model_type == "gpt-3.5-turbo":
            return "gpt-3.5-turbo"
        elif self.model_type == "llama":
            return "meta-llama/Llama-3.1-70B-Instruct"
        else:
            return self.model_type  # Return the Groq model name directly
    
    def complete(self, prompt, max_tokens=1024, temperature=0.2, top_p=0.95, stream=False, response_format=None, extra_body=None):
        """
        Generate a completion for the given prompt.
        
        Args:
            prompt (str): The prompt text
            max_tokens (int): Maximum tokens in response
            temperature (float): Randomness parameter
            top_p (float): Nucleus sampling parameter
            stream (bool): Whether to stream the response
            response_format (dict): Format for the response (e.g., {"type": "json_object"})
            extra_body (dict): Additional parameters for the API call
            
        Returns:
            str: Generated text
        """
        response_text = ""
        
        # Set default extra_body if not provided
        if extra_body is None:
            extra_body = {"service_tier": "on_demand"}
        
        if "gpt" in self.model_type.lower():
            # OpenAI API
            params = {
                "model": self.model_type,
                "messages": [{"role": "user", "content": prompt}],
                "max_tokens": max_tokens,
                "temperature": temperature,
                "top_p": top_p,
                "stream": stream
            }
            
            # Add response_format if specified
            if response_format:
                params["response_format"] = response_format
                
            # Add extra_body parameters
            if extra_body:
                params.update(extra_body)
                
            response = self.client.chat.completions.create(**params)
            response_text = response.choices[0].message.content.strip()
        
        elif self.model_type == "llama":
            # Hugging Face API
            model_name = self.get_model_name()
            params = {
                "model": model_name,
                "messages": [{"role": "user", "content": prompt}],
                "max_tokens": max_tokens,
                "temperature": temperature,
                "top_p": top_p,
                "stream": stream
            }
            
            # Add response_format if specified
            if response_format:
                params["response_format"] = response_format
                
            # Add extra_body parameters
            if extra_body:
                params.update(extra_body)
                
            response = self.client.chat.completions.create(**params)
            response_text = response.choices[0].message.content.strip()
        
        elif any(name in self.model_type.lower() for name in ["deepseek", "qwen"]) or (
            "llama" in self.model_type.lower() and self.model_type != "llama"
        ):
            # Groq API
            params = {
                "model": self.model_type,
                "messages": [{"role": "user", "content": prompt}],
                "max_tokens": max_tokens,
                "temperature": temperature,
                "top_p": top_p,
                "stream": stream
            }
            
            # Add response_format if specified
            if response_format:
                params["response_format"] = response_format
                
            # Add extra_body parameters
            if extra_body:
                params.update(extra_body)
                
            response = self.client.chat.completions.create(**params)
            response_text = response.choices[0].message.content.strip()
        
        else:
            # Default to OpenAI
            params = {
                "model": "gpt-3.5-turbo",
                "messages": [{"role": "user", "content": prompt}],
                "max_tokens": max_tokens,
                "temperature": temperature,
                "top_p": top_p,
                "stream": stream
            }
            
            # Add response_format if specified
            if response_format:
                params["response_format"] = response_format
                
            # Add extra_body parameters
            if extra_body:
                params.update(extra_body)
                
            response = self.client.chat.completions.create(**params)
            response_text = response.choices[0].message.content.strip()
        
        # Remove think tags from response
        response_text = remove_think_tags(response_text)
        
        return response_text


def extract_paragraph(text, section_name, client, model):
    """
    Extract a specific section from text using LLM.
    
    Args:
        text (str): Input text to process
        section_name (str): Name of section to extract
        client: LLMClient object (not the underlying client)
        model: Model name
    
    Returns:
        str: Extracted paragraph
    """
    prompt = f"""
    You are a helpful assistant that specializes in extracting specific sections from medical device documentation.
    
    Given the text below, please extract the {section_name.upper()} section.
    
    Return ONLY the text that describes the {section_name} with no additional commentary.
    If you cannot find the {section_name} section, respond with "No {section_name} information found."
    
    Text:
    {text[:500000]}  # Truncate to avoid token limits
    """
    
    return client.complete(
        prompt, 
        max_tokens=1024, 
        temperature=0.2,
        top_p=0.95,
        stream=False,
        response_format=None,  # Not returning JSON
        extra_body={"service_tier": "on_demand"}
    )


def write_sim_diff_discussion(subject_text, predicate_text, sim_or_diff, client, model):
    """
    Generate similarities or differences between two medical devices.
    
    Args:
        subject_text (str): Text about first device
        predicate_text (str): Text about second device
        sim_or_diff (str): Either "similarities" or "differences"
        client: LLMClient object (not the underlying client)
        model: Model name
    
    Returns:
        str: Generated analysis
    """
    prompt = f"""
    You are a medical device regulatory expert who specializes in 510(k) submissions to the FDA.
    
    You need to analyze the {sim_or_diff} between a subject device and a predicate device.
    
    Subject Device Information:
    {subject_text[:100000]}  # Truncate to avoid token limits
    
    Predicate Device Information:
    {predicate_text[:400000]}  # Truncate to avoid token limits
    
    Please write a comprehensive analysis of the {sim_or_diff} between these two medical devices, considering:
    - Indications for use
    - Device design and components
    - Operating principles
    - Technical specifications
    - Materials used
    - Performance characteristics
    
    Your response should be detailed yet concise, focusing only on substantial {sim_or_diff}. Format as paragraphs with clear subtopics.
    """
    
    return client.complete(
        prompt, 
        max_tokens=2000, 
        temperature=0.3,
        top_p=0.95,
        stream=False,
        response_format=None,  # Not returning JSON
        extra_body={"service_tier": "on_demand"}
    )


def write_detailed_sim_diff_discussion(kletter_text, predicate_manuals_text, user_manuals_text, client, model):
    """
    Generate a detailed comparison between devices using manuals.
    
    Args:
        kletter_text (str): K-letter text
        predicate_manuals_text (str): Predicate device manuals text
        user_manuals_text (str): User device manuals text
        client: LLMClient object (not the underlying client)
        model: Model name
    
    Returns:
        tuple: (similarities, differences)
    """
    # First prompt to analyze similarities
    similarities_prompt = f"""
    You are a medical device regulatory expert specializing in 510(k) submissions.
    
    Based on the following information, provide a detailed analysis of the SIMILARITIES between the subject device and predicate device.
    
    K-Letter Summary:
    {kletter_text[:100000]}
    
    Predicate Device Manuals:
    {predicate_manuals_text[:200000]}
    
    Subject Device Manuals:
    {user_manuals_text[:200000]}
    
    Focus on the following aspects in your analysis:
    1. Intended use and indications for use
    2. Technological characteristics
    3. Performance characteristics
    4. Operating principles
    5. Materials and components
    6. Safety features
    
    Provide a detailed, well-structured analysis of the significant similarities. Format your response as detailed paragraphs organized by the categories above.
    """
    
    # Second prompt to analyze differences
    differences_prompt = f"""
    You are a medical device regulatory expert specializing in 510(k) submissions.
    
    Based on the following information, provide a detailed analysis of the DIFFERENCES between the subject device and predicate device.
    
    K-Letter Summary:
    {kletter_text[:100000]}
    
    Predicate Device Manuals:
    {predicate_manuals_text[:200000]}
    
    Subject Device Manuals:
    {user_manuals_text[:200000]}
    
    Focus on the following aspects in your analysis:
    1. Intended use and indications for use
    2. Technological characteristics
    3. Performance characteristics
    4. Operating principles
    5. Materials and components
    6. Safety features
    
    Provide a detailed, well-structured analysis of the significant differences. Format your response as detailed paragraphs organized by the categories above.
    """
    
    similarities = client.complete(
        similarities_prompt, 
        max_tokens=2000, 
        temperature=0.3,
        top_p=0.95,
        stream=False,
        response_format=None,  # Not returning JSON
        extra_body={"service_tier": "on_demand"}
    )
    
    differences = client.complete(
        differences_prompt, 
        max_tokens=5000, 
        temperature=0.3,
        top_p=0.95,
        stream=False,
        response_format=None,  # Not returning JSON
        extra_body={"service_tier": "on_demand"}
    )
    
    return similarities, differences


def get_device_attributes(text, attributes_dict, client, model):
    """
    Extract specific device attributes from text.
    
    Args:
        text (str): Input text to process
        attributes_dict (dict): Dictionary of attributes to extract
        client: LLMClient object (not the underlying client)
        model: Model name
    
    Returns:
        dict: Extracted attributes
    """
    attrs_str = "\n".join([f"- {key}: {value}" for key, value in attributes_dict.items()])
    
    prompt = f"""
    You are a medical device expert. Extract the values for the following attributes from the text.
    
    Attributes to extract:
    {attrs_str}
    
    For each attribute, find the most relevant information in the text.
    Return your answer as a JSON dictionary with the attribute names as keys and the extracted values as values.
    If you cannot find information for an attribute, use "Not specified" as the value.
    
    Text:
    {text[:100000]}  # Truncate to avoid token limits
    """
    
    response = client.complete(
        prompt, 
        max_tokens=1500, 
        temperature=0.2,
        top_p=0.95,
        stream=False,
        response_format={"type": "json_object"},  # JSON response
        extra_body={"service_tier": "on_demand"}
    )
    
    # Extract JSON from response (simple approach)
    try:
        import json
        start_idx = response.find('{')
        end_idx = response.rfind('}') + 1
        if start_idx != -1 and end_idx != -1:
            json_str = response[start_idx:end_idx]
            return json.loads(json_str)
        else:
            return {}
    except:
        return {} 