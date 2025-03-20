from pydantic import BaseModel, ValidationError
from typing import Dict, List, Union, Tuple, Optional
import datetime


class DetailDict(BaseModel):
    details: Dict[str, str]


class DeviceMetadata(BaseModel):
    created_at: str = datetime.datetime.now().isoformat()
    updated_at: str = datetime.datetime.now().isoformat()
    version: str = "2.0.0"


class DeviceResult(BaseModel):
    k_number: str
    search_similarity_score: float
    indications: str
    device_description: str
    operating_principle: str
    device_details: DetailDict
    differences: str
    similarities: str
    metadata: DeviceMetadata
    device_name: str
    regulation_number: str
    regulation_name: str
    regulatory_class: str
    product_code: str
    received_date: Union[str, datetime.datetime]


class PredicateRequest(BaseModel):
    number_of_predicates: int
    indications: str
    device_details: str
    operating_principle: str
    device_attributes: Dict[str, str] = {}  # Optional field with default empty dict


class PredicateResponse(BaseModel):
    details: DetailDict
    results: List[DeviceResult]


class SimDiffRequest(BaseModel):
    knumber: str
    indications: str
    device_details: str
    operating_principle: str


class SimDiffResponse(BaseModel):
    similarities: str
    differences: str


class DetailedSimDiffRequest(BaseModel):
    knumber: str
    predicate_manuals_s3_path: str
    user_device_manuals_s3_path: str


class IndicsDevDetailsOpPrincipal(BaseModel):
    indications: str
    device_details: str
    operating_principle: str


class PredicateKnumberResponse(BaseModel):
    results: List[Tuple[str, float]]  # List of (knumber, score) tuples


class PredicateDetailsRequest(BaseModel):
    knumbers: List[str]


class PredicateDetails(BaseModel):
    k_number: str
    device_description: str
    operating_principle: str
    indications: str
    device_name: str
    regulation_number: str
    regulation_name: str
    regulatory_class: str
    product_code: str
    received_date: Union[str, datetime.datetime]
    submitter: str
    manufacturer: str


class PredicateDetailsResponse(BaseModel):
    results: List[PredicateDetails]


class KeywordSynonyms(BaseModel):
    """Model representing a keyword and its synonyms"""
    keyword: str
    synonyms: List[str]


class TextAnalysis(BaseModel):
    """Model representing analysis of a text section with keywords and synonyms"""
    text_type: str  # "indications", "device_details", or "operating_principle"
    keywords_synonyms: List[KeywordSynonyms]


class EnhancedPredicateResponse(BaseModel):
    """Enhanced response model that includes text analysis with keywords and synonyms"""
    details: DetailDict
    text_analysis: List[TextAnalysis]
    results: List[DeviceResult] 