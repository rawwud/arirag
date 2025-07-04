"""
Al Mujtama News Assistant - FastAPI Application
A chatbot for the Al Mujtama news magazine with RAG capabilities using Pinecone and Groq.
"""

import json
import requests
# import numpy as np  # Removed to reduce package size for Vercel
import re
import logging
import os
import time # Added for optional sleep after index creation
from groq import Groq
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel
from pinecone import Pinecone
from pinecone import PineconeApiException
from typing import Optional
from together import Together
# from together import Together  # Removed to reduce package size

# TOG_API_KEY = "4e4f7d38e1f953da9cfd545a6bab84509a52fc4a68e7c68a876eaf9373827e2a"
# client_tog = Together(api_key=TOG_API_KEY)  # Removed to reduce package size

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="Al Mujtama News Assistant",
    description="Arabic news chatbot with RAG capabilities",
    version="1.0.0"
)

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount static files
app.mount("/static", StaticFiles(directory="."), name="static")

# Initialize templating
templates = Jinja2Templates(directory=".")

# ============================================================================
# CONFIGURATION AND SETUP
# ============================================================================

# Configuration with environment variable support
JINA_API_KEY = os.getenv("JINA_API_KEY", "jina_2ea8da0cd64742f2a49cb2acd811a173xwOvb7Tf0TYVfmiSOeFgN0Pow2wm")
GROQ_API_KEY = os.getenv("GROQ_API_KEY", "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")
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY", "pcsk_LQ2ZS_BVEVWZhyeL5yGS92fTJc2sAbqqpvC7MVdXjg49efGP6AnUQPe26hpPVggwJaJUe")
INDEX_NAME = os.getenv("PINECONE_INDEX_NAME", "testerbedtheone")
EMBEDDING_DIM = int(os.getenv("EMBEDDING_DIM", "3072"))
TOG_API_KEY = os.getenv("TOG_API_KEY", "240f6c9b6abae24fe96f5be23060e38f76b89fd31c121f807615a3dd90385278")

# Initialize Groq Client
# Global variables
documents = []
pc = None
index = None
pinecone_available = False
groq_client = None
tog_client = None

# ============================================================================
# CLIENT INITIALIZATION
# ============================================================================

def initialize_groq_client():
    """Initialize Groq client with error handling"""
    global groq_client
    try:
        if groq_client is None:
            groq_client = Groq(api_key=GROQ_API_KEY)
            logger.info("Groq client initialized successfully")
        return groq_client
    except Exception as e:
        logger.error(f"Error initializing Groq client: {str(e)}")
        return None

def get_groq_client():
    """Get Groq client, initializing if needed"""
    if groq_client is None:
        return initialize_groq_client()
    return groq_client

def initialize_tog_client():
    """Initialize TOG client with error handling"""
    global tog_client
    try:
        if tog_client is None:
            tog_client = Together(api_key=TOG_API_KEY)
            logger.info("TOG client initialized successfully")
        return tog_client
    except Exception as e:
        logger.error(f"Error initializing TOG client: {str(e)}")
        return None

def get_tog_client():
    """Get TOG client, initializing if needed"""
    if tog_client is None:
        return initialize_tog_client()
    return tog_client

# ============================================================================
# DATA LOADING
# ============================================================================

def load_documents():
    """Load documents from JSON file if available"""
    global documents
    try:
        with open('extracted_articles_cleaned_three_one.json', 'r', encoding='utf-8') as f:
            documents = json.load(f)
        logger.info(f"Successfully loaded {len(documents)} documents from JSON file")
    except FileNotFoundError:
        logger.warning("Document file not found - running without local documents. Relying on Pinecone data.")
        documents = []
    except Exception as e:
        logger.error(f"Error loading documents: {str(e)} - continuing without local documents")
        documents = []

# ============================================================================
# EMBEDDING AND PINECONE FUNCTIONS
# ============================================================================

def get_embedding(text: str) -> list:
    """Convert text to vector using Jina AI embeddings"""
    try:
        url = "https://api.jina.ai/v1/embeddings"
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {JINA_API_KEY}"
        }
        data = {
            "input": text,
            "model": "jina-embeddings-v3"
        }

        response = requests.post(url, headers=headers, json=data)
        response.raise_for_status()
        result = response.json()

        if "data" in result and len(result["data"]) > 0 and "embedding" in result["data"][0]:
            base_embedding = result["data"][0]["embedding"]
            
            # Resize embedding to match Pinecone dimension
            if len(base_embedding) < EMBEDDING_DIM:
                padding = [0.0] * (EMBEDDING_DIM - len(base_embedding))
                return base_embedding + padding
            elif len(base_embedding) > EMBEDDING_DIM:
                return base_embedding[:EMBEDDING_DIM]
            else:
                return base_embedding
        else:
            raise ValueError(f"Unexpected API response format: {result}")

    except Exception as e:
        logger.error(f"Error getting embedding: {str(e)}")
        raise

def initialize_pinecone() -> bool:
    """Initialize Pinecone connection"""
    global pc, index, pinecone_available
    
    try:
        pc = Pinecone(api_key=PINECONE_API_KEY)
        
        # Create index if it doesn't exist
        existing_indexes = [index_info.name for index_info in pc.list_indexes()]
        if INDEX_NAME not in existing_indexes:
            logger.info(f"Creating Pinecone index '{INDEX_NAME}'...")
            pc.create_index(
                name=INDEX_NAME,
                dimension=EMBEDDING_DIM,
                metric="cosine",
                spec={
                    "serverless": {
                        "cloud": "aws",
                        "region": "us-east-1"
                    }
                }
            )
            time.sleep(2)  # Wait for index creation
            
        index = pc.Index(INDEX_NAME)
        logger.info(f"Successfully connected to Pinecone index '{INDEX_NAME}'")
        
        # Check index status
        stats = index.describe_index_stats()
        vector_count = stats.get('total_vector_count', 0)
        logger.info(f"Pinecone index contains {vector_count} vectors")
        
        pinecone_available = True
        return True
        
    except Exception as e:
        logger.error(f"Error initializing Pinecone: {str(e)}")
        pinecone_available = False
        return False

def upsert_documents():
    """Upload documents to Pinecone"""
    if not documents or not pinecone_available:
        logger.warning("No documents or Pinecone unavailable for upsert")
        return
        
    try:
        BATCH_SIZE = 1
        upsert_data = []
        total_batches = 0

        for i, doc in enumerate(documents):
            title = doc.get("title", "No Title")
            content = doc.get("body", "")
            full_context = f"Title: {title}\nBody: {content}"

            try:
                emb = get_embedding(full_context)
                if len(emb) != EMBEDDING_DIM:
                    logger.warning(f"Embedding dimension mismatch for doc {i}")
                    continue

                upsert_data.append((str(i), emb, {"title": title, "body": content}))

                if len(upsert_data) >= BATCH_SIZE or i == len(documents) - 1:
                    if upsert_data:
                        logger.info(f"Upserting batch {total_batches + 1}")
                        index.upsert(vectors=upsert_data)
                        total_batches += 1
                        upsert_data = []

            except Exception as e:
                logger.error(f"Error processing document {i}: {str(e)}")
                continue

        logger.info(f"Completed upserting {total_batches} batches")
        
    except Exception as e:
        logger.error(f"Error in upsert_documents: {str(e)}")
        raise

def retrieve_documents(query: str, top_k: int = 7) -> tuple:
    """Retrieve relevant documents from Pinecone"""
    try:
        if not pinecone_available or index is None:
            logger.warning("Pinecone not available")
            return [], []
            
        query_vector = get_embedding(query)
        query_response = index.query(
            vector=query_vector,
            top_k=top_k,
            include_metadata=True
        )

        results = query_response.get("matches", [])
        titles = [match["metadata"]["title"] for match in results]
        contents = [match["metadata"]["body"] for match in results]
        scores = [match["score"] for match in results]

        logger.info(f"Retrieved {len(titles)} documents with scores: {scores}")
        return titles, contents

    except Exception as e:
        logger.error(f"Error retrieving documents: {str(e)}")
        return [], []

# ============================================================================
# AI PROCESSING FUNCTIONS
# ============================================================================

def remove_thoughts(text: str) -> str:
    """Remove thought process tags from text"""
    return re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL)

async def classify_query(query: str) -> str:
    """Classify if query needs documents (NS) or can be answered directly (safe)"""
    try:
        client = get_groq_client()
        if client is None:
            logger.error("Groq client not available")
            return "NS"  # Default to document retrieval
            
        completion = client.chat.completions.create(
            model="llama-3.1-8b-instant",
            messages=[
                {
                    "role": "system",
                    "content": """You are a query classifier for Al Mujtama news magazine. 
                    Reply only with "safe" or "NS".
                    - "safe": for greetings or general questions about the assistant
                    - "NS": for questions about news, articles, or Al Mujtama content"""
                },
                {"role": "user", "content": query}
            ],
            temperature=0.2,
            max_tokens=10,
        )

        response = completion.choices[0].message.content.strip()
        return "NS" if "NS" in response else "safe"

    except Exception as e:
        logger.error(f"Error classifying query: {str(e)}")
        return "NS"  # Default to document retrieval

async def safe_generate_answer(query: str) -> dict:
    """Generate answer without document retrieval"""
    try:
        client = get_groq_client()
        if client is None:
            return {
                "response": "عذراً، الخدمة غير متاحة حالياً. يرجى المحاولة مرة أخرى لاحقاً.",
                "reference": "",
                "error": "Groq client not available"
            }
            
        prompt = f"""You are an Arabic assistant for Al Mujtama news magazine.
        Respond in Arabic to this greeting or general question: {query}
        Keep responses friendly and brief."""

        response = client.chat.completions.create(
            model="meta-llama/llama-4-scout-17b-16e-instruct",
            messages=[
                {"role": "system", "content": "Always respond in Arabic, be helpful and professional."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.6,
            max_tokens=1024,
        )

        return {
            "response": remove_thoughts(response.choices[0].message.content),
            "reference": "general knowledge"
        }

    except Exception as e:
        logger.error(f"Error in safe generation: {str(e)}")
        return {
            "response": "عذراً، حدث خطأ. يرجى المحاولة مرة أخرى.",
            "reference": "",
            "error": str(e)
        }

async def NS_generate_answer(query: str) -> dict:
    """Generate answer with document retrieval"""
    try:
        client = get_tog_client()
        if client is None:
            return {
                "response": "عذراً، الخدمة غير متاحة حالياً. يرجى المحاولة مرة أخرى لاحقاً.",
                "reference": "",
                "error": "TOG client not available"
            }
            
        retrieved_titles, retrieved_contents = retrieve_documents(query, top_k=7)

        if not retrieved_titles:
            return {
                "response": "عذراً، لم أتمكن من العثور على معلومات ذات صلة في قاعدة البيانات.",
                "reference": ""
            }

        combined_context = ""
        for title, content in zip(retrieved_titles, retrieved_contents):
            combined_context += f"Document Title: {title}\nDocument Content: {content}\n\n"

        prompt = f"""You are an Arabic assistant for Al Mujtama magazine.
        Answer this question using only the provided documents: {query}
        
        Context from Al Mujtama database:
        {combined_context}
        
        Respond in Arabic, be concise and accurate."""

        response = client.chat.completions.create(
            model="meta-llama/Llama-4-Scout-17B-16E-Instruct",
            messages=[
                {"role": "system", "content": "Always respond in Arabic using only the provided context."},
                {"role": "user", "content": prompt}
            ],
            max_tokens=8000,
            temperature=0.6,
        )

        return {
            "response": remove_thoughts(response.choices[0].message.content),
            "reference": combined_context
        }

    except Exception as e:
        logger.error(f"Error in NS generation: {str(e)}")
        return {
            "response": "عذراً، حدث خطأ أثناء معالجة طلبك.",
            "reference": "",
            "error": str(e)
        }

# ============================================================================
# PYDANTIC MODELS
# ============================================================================

class ChatMessage(BaseModel):
    message: str

# ============================================================================
# API ROUTES
# ============================================================================

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {"status": "healthy", "message": "Al Mujtama Assistant is running"}

@app.get("/api/test")
async def test_endpoint():
    """Test endpoint"""
    return {
        "message": "API is working", 
        "endpoints": ["health", "api/test", "api/chat", "debug"]
    }

@app.get("/debug")
async def debug_status():
    """Debug status endpoint"""
    try:
        pinecone_vectors = 0
        pinecone_status = "unavailable"
        
        if pinecone_available and index is not None:
            try:
                stats = index.describe_index_stats()
                pinecone_vectors = stats.get('total_vector_count', 0)
                pinecone_status = "connected"
            except Exception as e:
                pinecone_status = f"error: {str(e)}"
        
        return {
            "status": "running",
            "documents_loaded": len(documents),
            "pinecone_available": pinecone_available,
            "pinecone_status": pinecone_status,
            "pinecone_vectors": pinecone_vectors,
            "message": "Debug endpoint working"
        }
    except Exception as e:
        return {"status": "error", "error": str(e)}

@app.post("/api/chat")
async def chat(chat_message: ChatMessage):
    """Main chat endpoint"""
    try:
        if not chat_message.message.strip():
            raise HTTPException(status_code=400, detail="No message provided")

        logger.info(f"Received chat request: {chat_message.message[:100]}...")

        # Check Pinecone availability
        if not pinecone_available:
            return JSONResponse(
                status_code=200,
                content={
                    "response": "عذراً، الخدمة غير متاحة حالياً. يرجى المحاولة مرة أخرى لاحقاً.",
                    "reference": "",
                    "error": "Pinecone service unavailable"
                }
            )

        # Classify query
        try:
            query_type = await classify_query(chat_message.message)
            logger.info(f"Query classified as: {query_type}")
        except Exception as e:
            logger.error(f"Error in query classification: {str(e)}")
            return JSONResponse(
                status_code=200,
                content={
                    "response": "عذراً، حدث خطأ في تصنيف الاستفسار.",
                    "reference": "",
                    "error": f"Classification error: {str(e)}"
                }
            )

        # Generate answer
        try:
            if query_type == "safe":
                answer_data = await safe_generate_answer(chat_message.message)
            else:
                answer_data = await NS_generate_answer(chat_message.message)
        except Exception as e:
            logger.error(f"Error in answer generation: {str(e)}")
            return JSONResponse(
                status_code=200,
                content={
                    "response": "عذراً، حدث خطأ أثناء إنتاج الإجابة.",
                    "reference": "",
                    "error": f"Answer generation error: {str(e)}"
                }
            )

        answer_data["query_type"] = query_type
        return answer_data

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Unexpected error in chat endpoint: {str(e)}")
        return JSONResponse(
            status_code=200,
            content={
                "response": "عذراً، حدث خطأ غير متوقع.",
                "reference": "",
                "error": f"Unexpected error: {str(e)}"
            }
        )

@app.post("/api/seed-pinecone")
async def seed_pinecone():
    """Manually seed Pinecone with documents"""
    try:
        if not pinecone_available:
            return {"status": "error", "message": "Pinecone not available"}
        
        if len(documents) == 0:
            return {"status": "error", "message": "No documents loaded to seed"}
        
        logger.info(f"Starting to seed Pinecone with {len(documents)} documents")
        upsert_documents()
        
        stats = index.describe_index_stats()
        vector_count = stats.get('total_vector_count', 0)
        
        return {
            "status": "success", 
            "message": f"Successfully seeded Pinecone with {len(documents)} documents",
            "total_vectors": vector_count
        }
    except Exception as e:
        logger.error(f"Error seeding Pinecone: {str(e)}")
        return {"status": "error", "message": str(e)}

@app.post("/api/reindex")
async def reindex():
    """Force reindexing of all documents"""
    try:
        if not pinecone_available:
            return {"status": "error", "message": "Pinecone not available"}
            
        logger.info("Starting reindexing process")
        index.delete(delete_all=True)
        upsert_documents()
        return {"status": "success", "message": "Reindexing completed"}
    except Exception as e:
        logger.error(f"Error during reindexing: {str(e)}")
        return {"status": "error", "message": str(e)}

# ============================================================================
# WEB ROUTES
# ============================================================================

@app.get("/", response_class=HTMLResponse)
async def read_root(request: Request):
    """Main page"""
    return templates.TemplateResponse("index.html", {"request": request})





@app.get("/api/documents")
async def get_documents():
    """Get documents info"""
    return {"message": "Direct document serving is disabled. Data is accessed via Pinecone."}


# ============================================================================
# STARTUP
# ============================================================================

@app.on_event("startup")
async def startup_event():
    """Application startup"""
    logger.info("Starting Al Mujtama Assistant...")
    
    # Initialize Groq client
    if initialize_groq_client():
        logger.info("Groq client initialized successfully")
    else:
        logger.warning("Groq client initialization failed")
    
    # Load documents
    load_documents()
    
    # Initialize Pinecone
    if initialize_pinecone():
        logger.info("Application startup completed successfully")
    else:
        logger.warning("Application started but Pinecone initialization failed")

# ============================================================================
# MAIN
# ============================================================================

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
