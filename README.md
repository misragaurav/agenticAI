# agenticAI

An intelligent, AI-powered system for searching and analyzing medical device predicates for FDA 510(k) submissions. This API helps medical device manufacturers find appropriate predicate devices and analyze similarities and differences.

## Features

- LLM-assisted semantic search for medical device predicates
- Keyword extraction and synonym generation for enhanced search
- Weighted search prioritizing indications for use
- Text analysis and visualization of keywords and synonyms
- Detailed predicate device information retrieval
- Similarities and differences analysis between devices

## Configuration

Key settings are centralized in the `app/core/config.py` file:

- **Keyword Extraction**: Controls the number of keywords and synonyms
  - `KEYWORD_EXTRACTION_COUNT`: Number of keywords to extract (default: 4)
  - `SYNONYM_COUNT_PER_KEYWORD`: Number of synonyms per keyword (default: 3)

- **Search Weights**: Controls the importance of different components
  - `SEARCH_WEIGHTS["indications"]`: Weight for indications matches (70%)
  - `SEARCH_WEIGHTS["device_details"]`: Weight for device details matches (15%)
  - `SEARCH_WEIGHTS["operating_principle"]`: Weight for operating principle matches (15%)

## Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/agenticAI.git
cd agenticAI

# Create and activate virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

## Usage

```bash
# Start the FastAPI server
cd app
uvicorn main:app --reload
```

Visit `http://localhost:8000/docs` to access the Swagger UI documentation.

## Project Structure

```
app/
├── auth/               # Authorization modules
│   └── aws/            # AWS Cognito JWT authorization
├── core/               # Core functionality
│   └── config.py       # Configuration settings
├── models/             # Data models
│   └── schemas.py      # Pydantic schemas
├── routers/            # API routes
│   ├── pdf.py          # PDF-related routes
│   ├── predicates.py   # Predicate device routes
│   └── similarities.py # Similarity/difference analysis routes
├── services/           # Business logic services
│   ├── llm.py          # Language model service
│   ├── mongodb.py      # MongoDB service
│   └── search.py       # Vector search service
├── utils/              # Utility functions
│   ├── embeddings.py   # Text embedding utilities
│   └── pdf.py          # PDF processing
└── main.py             # Application entry point
```

## Dependencies

- FastAPI: Web framework
- PyMongo: MongoDB integration
- AWS Bedrock: Text embeddings
- LLM Services: OpenAI GPT and Groq (DeepSeek, Llama)
- PyMuPDF: PDF processing
- AWS Cognito: Authentication

## API Endpoints

- **GET /** - Welcome message
- **POST /getPredicates** - Find similar predicate devices
- **POST /getSimilarityDifferences** - Generate similarity and difference analyses
- **POST /getDetailedSimDiffs** - Generate detailed analyses with device manuals
- **GET /getPDF/{knumber}** - Retrieve a K-letter PDF
- **POST /getPredicateKnumbers** - Get a list of similar K-numbers
- **POST /getPredicateDetails** - Get detailed information for specific K-numbers 