# AWS Bedrock Knowledge Base Chat with Gradio UI

This application provides a user-friendly web interface for interacting with AWS Bedrock Knowledge Bases using various AI models.

## Features

- Interactive web UI for knowledge base chat using Gradio
- Support for additional keywords to enhance search
- Support for multiple AI models (Claude, Titan, DeepSeek)
- Adjustable settings for retrieval and generation parameters
- Conversation history with copy functionality

## Setup and Installation

1. Make sure you have the required Python packages:
   ```bash
   pip install gradio boto3
   ```

2. Configure your AWS credentials:
   ```bash
   aws configure
   ```

## Usage

### Basic Usage

```bash
python bedrock_kb_chat_gradio.py
```

This will start the Gradio web interface with default settings:
- Knowledge Base ID: UAA8CIQ6UZ
- Model: anthropic.claude-3-5-haiku-20241022-v1:0
- AWS Region: us-west-2
- Port: Automatically selected available port

### Advanced Usage

You can customize the application by providing command-line arguments:

```bash
python bedrock_kb_chat_gradio.py \
  --kb-id YOUR_KB_ID \
  --model-id anthropic.claude-3-sonnet-20240229-v1:0 \
  --region us-west-2 \
  --num-results 15 \
  --temperature 0.1 \
  --max-tokens 1500 \
  --server-port 7878 \
  --share
```

### Command-line Arguments

- `--kb-id`: Knowledge base ID (default: UAA8CIQ6UZ)
- `--model-id`: Model ID to use for chat (default: anthropic.claude-3-5-haiku-20241022-v1:0)
- `--region`: AWS region (default: us-west-2)
- `--profile`: AWS profile name (optional)
- `--num-results`: Number of source documents to retrieve (default: 20)
- `--temperature`: Temperature for generation (default: 0.1)
- `--max-tokens`: Maximum tokens to generate (default: 1000)
- `--share`: Create a shareable link for the Gradio interface
- `--server-port`: Port to run the server on (default: None - automatically selects an available port)

## Using the Interface

1. Enter your question in the "Question" text box
2. Optionally add keywords in the "Keywords" text box to enhance the search
3. Click "Submit" or press Enter to get a response
4. Use "Clear Conversation" to start a new chat session
5. Adjust advanced settings like number of documents to retrieve and temperature as needed

## Examples

### Basic Question
Just enter your question in the main text box:
```
What are the main features of AWS Bedrock?
```

### Question with Keywords
Include specific keywords to enhance the search:
```
Question: How does vector search work in AWS Bedrock?
Keywords: embeddings, similarity, vector database, performance
```

## Requirements

- Python 3.7+
- gradio
- boto3
- AWS account with configured credentials
- Access to AWS Bedrock Knowledge Bases 