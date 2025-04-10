#!/usr/bin/env python

import gradio as gr
import boto3
import json
from typing import Dict, List, Optional, Union, Any
import argparse
import time
from bedrock_kb_chat import BedrockKnowledgeBaseChat

def create_chat_interface(
    kb_id: str = "HGVFQQ2VFF",
    model_id: str = "anthropic.claude-3-5-haiku-20241022-v1:0",
    region_name: str = "us-west-2",
    profile_name: Optional[str] = None,
    num_results: int = 20,
    temperature: float = 0.1,
    max_tokens: int = 1000,
):
    """Create a Gradio interface for the Bedrock Knowledge Base chat application"""
    
    # Initialize the chat client
    chat_client = BedrockKnowledgeBaseChat(
        kb_id=kb_id,
        model_id=model_id,
        region_name=region_name,
        profile_name=profile_name,
    )
    
    # Store conversation history and source metadata
    conversation_state = {"history": [], "current_sources": []}
    
    def chat_with_kb(message: str, keywords: str, history: List[List[str]] = None):
        """Process the chat message with the keywords"""
        
        # Combine message with keywords if provided
        if keywords.strip():
            combined_message = f"{message}\nKeywords: {keywords}"
        else:
            combined_message = message
            
        # Call the knowledge base chat
        inference_params = {
            "temperature": temperature,
        }
        
        # Adjust parameters based on model
        if "claude" in model_id.lower():
            inference_params["max_tokens"] = max_tokens
        elif "deepseek" in model_id.lower():
            inference_params["max_tokens"] = max_tokens
        else:
            inference_params["maxTokenCount"] = max_tokens
            
        # Get response from KB
        result = chat_client.chat(
            query=combined_message,
            num_results=num_results,
            inference_params=inference_params,
            streaming=False,
        )
        
        # Extract response and source documents
        response = result['response']
        source_documents = result.get('source_documents', [])
        
        # Store source documents for display
        conversation_state["current_sources"] = source_documents
        
        # Update conversation state with messages format
        if not history:
            history = []
        
        # Add the new message in the messages format
        history.append({"role": "user", "content": message})
        history.append({"role": "assistant", "content": response})
        conversation_state["history"] = history
        
        return history
    
    def clear_conversation():
        """Clear the conversation history and source metadata"""
        conversation_state["history"] = []
        conversation_state["current_sources"] = []
        return [], "<p>Source documents will appear here after a query.</p>"
    
    def format_source_documents():
        """Format source documents with metadata for display"""
        source_documents = conversation_state["current_sources"]
        if not source_documents:
            return []
        
        formatted_sources = []
        for i, doc in enumerate(source_documents):
            # Extract basic metadata
            score = doc['metadata'].get('score', 'N/A')
            source = doc['metadata'].get('source', 'N/A')
            
            # Format source with all metadata
            source_info = f"<b>Source {i+1}</b> (Score: {score:.4f})<br>"
            source_info += f"<b>Source:</b> {source}<br>"
            
            # Add all other metadata
            for key, value in doc['metadata'].items():
                if key not in ["score", "source"]:
                    source_info += f"<b>{key}:</b> {value}<br>"
            
            # Add the content snippet
            source_info += f"<br><b>Content:</b><br>{doc['content'][:500]}..."
            formatted_sources.append(source_info)
        
        return formatted_sources
    
    # Create the Gradio interface
    with gr.Blocks(title="Knowledge Base Chat") as demo:
        gr.Markdown(f"""
        # Knowledge Base Chat
        
        **Knowledge Base ID:** {kb_id}  
        **Model:** {model_id}  
        **Region:** {region_name}
        """)
        
        with gr.Row():
            with gr.Column(scale=4):
                chatbot = gr.Chatbot(
                    label="Conversation",
                    height=500,
                    show_copy_button=True,
                    type="messages",
                )
            with gr.Column(scale=2):
                sources_html = gr.HTML(
                    label="Source Documents",
                    value="<p>Source documents will appear here after a query.</p>",
                )
            
        with gr.Row():
            with gr.Column(scale=4):
                msg = gr.Textbox(
                    placeholder="Enter your question or prompt here...",
                    label="Question",
                    lines=2,
                )
            
        with gr.Row():
            with gr.Column(scale=4):
                keywords = gr.Textbox(
                    placeholder="Optional: Add keywords to enhance the search (comma separated)",
                    label="Keywords to enhance the search",
                    lines=1,
                )
        
        with gr.Row():
            with gr.Column(scale=1):
                submit_btn = gr.Button("Submit", variant="primary")
            with gr.Column(scale=1):
                clear_btn = gr.Button("Clear Conversation")
                
        with gr.Accordion("Advanced Settings", open=False):
            num_results_slider = gr.Slider(
                minimum=1, 
                maximum=50, 
                value=num_results, 
                step=1, 
                label="Number of documents to retrieve",
                info="Higher values may improve answer quality but increase response time"
            )
            temperature_slider = gr.Slider(
                minimum=0.0, 
                maximum=1.0, 
                value=temperature, 
                step=0.1, 
                label="Temperature",
                info="Higher values make output more random, lower values more deterministic"
            )
            
        # Function to update sources display
        def update_sources():
            formatted_sources = format_source_documents()
            if not formatted_sources:
                return "<p>No sources available for this query.</p>"
                
            # Join all formatted sources with horizontal separators
            return "<hr>".join(formatted_sources)
        
        # Set up event handlers
        def on_submit(message, keywords, chatbot):
            history = chat_with_kb(message, keywords, chatbot)
            sources_html = update_sources()
            return history, "", sources_html
        
        submit_btn.click(
            on_submit,
            inputs=[msg, keywords, chatbot],
            outputs=[chatbot, msg, sources_html],
        )
        
        msg.submit(
            on_submit,
            inputs=[msg, keywords, chatbot],
            outputs=[chatbot, msg, sources_html],
        )
        
        clear_btn.click(
            clear_conversation,
            outputs=[chatbot, sources_html],
        )
        
        # Update parameters when sliders change
        def update_num_results(value):
            nonlocal num_results
            num_results = value
            
        def update_temperature(value):
            nonlocal temperature
            temperature = value
            
        num_results_slider.change(
            update_num_results,
            inputs=[num_results_slider],
        )
        
        temperature_slider.change(
            update_temperature,
            inputs=[temperature_slider],
        )
    
    return demo

def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description="Gradio UI for AWS Bedrock Knowledge Base Chat")
    parser.add_argument("--kb-id", default="HGVFQQ2VFF", help="Knowledge base ID for Fixed-Chunking")
    parser.add_argument("--model-id", default="anthropic.claude-3-5-haiku-20241022-v1:0", help="Model ID to use for chat")
    parser.add_argument("--region", default="us-west-2", help="AWS region")
    parser.add_argument("--profile", default=None, help="AWS profile")
    parser.add_argument("--num-results", type=int, default=20, help="Number of source documents to retrieve")
    parser.add_argument("--temperature", type=float, default=0.1, help="Temperature for generation")
    parser.add_argument("--max-tokens", type=int, default=1000, help="Maximum tokens to generate")
    parser.add_argument("--share", action="store_true", help="Create a shareable link")
    parser.add_argument("--server-port", type=int, default=None, help="Port to run the server on (None for auto-select)")
    
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()
    
    # Create and launch the interface
    demo = create_chat_interface(
        kb_id=args.kb_id,
        model_id=args.model_id,
        region_name=args.region,
        profile_name=args.profile,
        num_results=args.num_results,
        temperature=args.temperature,
        max_tokens=args.max_tokens,
    )
    
    # Launch the app
    demo.launch(
        pwa=True,
        share=True,
        server_port=None,  # Use any available port
        server_name="0.0.0.0"  # Bind to all interfaces
    ) 