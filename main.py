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
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel
from pinecone import Pinecone, ServerlessSpec
from typing import Optional
# from together import Together  # Removed to reduce package size

# TOG_API_KEY = "4e4f7d38e1f953da9cfd545a6bab84509a52fc4a68e7c68a876eaf9373827e2a"
# client_tog = Together(api_key=TOG_API_KEY)  # Removed to reduce package size

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

app = FastAPI()

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize templating
templates = Jinja2Templates(directory="templates")

# ---------------------- Configuration ----------------------
JINA_API_KEY = "jina_0e8359e715f24fef84a308b25dd08678MDMNGM42kSPBvmP3cM7HciTovCJy"
GROQ_API_KEY = "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"
PINECONE_API_KEY = "pcsk_LQ2ZS_BVEVWZhyeL5yGS92fTJc2sAbqqpvC7MVdXjg49efGP6AnUQPe26hpPVggwJaJUe"
PINECONE_ENV = "us-east1-gcp"
INDEX_NAME = "testerbedtheone"
EMBEDDING_DIM = 3072  # Match your Pinecone index dimension

# Initialize Groq Client
groq_client = Groq(api_key=GROQ_API_KEY)

# Load Company Documents (re-enabled for demo)
documents = []
try:
    # Try to load documents if the file exists
    with open('extracted_articles_cleaned_three_one.json', 'r', encoding='utf-8') as f:
        documents = json.load(f)
    logger.info(f"Successfully loaded {len(documents)} documents from JSON file")
except FileNotFoundError:
    logger.warning("Document file not found - running without local documents. Relying on Pinecone data.")
    documents = []
except Exception as e:
    logger.error(f"Error loading documents: {str(e)} - continuing without local documents")
    documents = []


def get_embedding(text):
    """
    Converts text into a vector using Jina AI embeddings API with error handling
    """
    try:
        url = "https://api.jina.ai/v1/embeddings"
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {JINA_API_KEY}"
        }

        # For Jina embeddings-v3, we use their default dimension and then resize
        # The model parameter should be jina-embeddings-v2-base-en or jina-embeddings-v3
        data = {
            "input": text,  # Directly pass the text string, not a list with dict
            "model": "jina-embeddings-v3"
        }

        response = requests.post(url, headers=headers, json=data)
        response.raise_for_status()

        result = response.json()
        logger.debug(f"Jina API response received for text: {text[:50]}...")

        # Extract the embedding
        if "data" in result and len(result["data"]) > 0 and "embedding" in result["data"][0]:
            # Get the base embedding
            base_embedding = result["data"][0]["embedding"]

            # If Pinecone expects 1536 dimensions but Jina gives a different size,
            # We need to resize it to match
            if len(base_embedding) < EMBEDDING_DIM:
                # Pad with zeros to reach EMBEDDING_DIM
                padding = [0.0] * (EMBEDDING_DIM - len(base_embedding))
                return base_embedding + padding
            elif len(base_embedding) > EMBEDDING_DIM:
                # Truncate to EMBEDDING_DIM
                return base_embedding[:EMBEDDING_DIM]
            else:
                return base_embedding
        else:
            raise ValueError(f"Unexpected API response format: {result}")

    except Exception as e:
        logger.error(f"Error getting embedding: {str(e)}")
        raise


# Initialize Pinecone with better error handling
pc = None
index = None
pinecone_available = False # Initialize as False here

def initialize_pinecone():
    """
    Initialize Pinecone connection and perform initial seeding if needed.
    This function ensures the index is created (if it doesn't exist)
    and populates it with documents ONLY IF the index is empty.
    """
    global pc, index, pinecone_available # Ensure these are updated globally
    try:
        pc = Pinecone(api_key=PINECONE_API_KEY)
        
        # Check if index exists
        index_exists = INDEX_NAME in pc.list_indexes().names()

        if not index_exists:
            logger.info(f"Pinecone index '{INDEX_NAME}' not found, creating it...")
            pc.create_index(
                name=INDEX_NAME,
                dimension=EMBEDDING_DIM,
                metric="cosine",
                spec=ServerlessSpec(cloud="aws", region="us-east-1")
            )
            # Give Pinecone a moment to create the index (optional, but can help)
            time.sleep(1) 
            
        # Connect to the Pinecone index (whether newly created or existing)
        index = pc.Index(INDEX_NAME) 
        logger.info(f"Successfully connected to Pinecone index '{INDEX_NAME}'")

        # Check if the index is empty AND if local documents are loaded
        stats = index.describe_index_stats()
        vector_count = stats.get('total_vector_count', 0)
        
        if vector_count == 0 and documents: # Condition to trigger an upload
            logger.warning(f"Pinecone index '{INDEX_NAME}' is empty but local documents exist ({len(documents)}). Attempting initial seeding...")
            upsert_documents() # Call the upsert function to populate Pinecone
            logger.info("Initial seeding of Pinecone completed.")
            # Re-check stats after seeding
            stats = index.describe_index_stats()
            vector_count = stats.get('total_vector_count', 0)
            logger.info(f"Pinecone index now contains {vector_count} vectors.")
        elif vector_count > 0: # Condition for direct connection (no upload needed)
            logger.info(f"Pinecone index '{INDEX_NAME}' already contains {vector_count} vectors. Skipping initial seeding.")
        else: # Pinecone connected, but no local docs to seed even if empty
            logger.info("Pinecone index initialized but no local documents were found to seed.")

        pinecone_available = True # Set the global flag to indicate Pinecone is ready
        return True
    except Exception as e:
        logger.error(f"Error initializing Pinecone: {str(e)}")
        pinecone_available = False # Set the global flag to indicate Pinecone is not ready
        return False

# This line calls the initialization function when the application starts
pinecone_available = initialize_pinecone()


def upsert_documents():
    """
    Upserts documents to Pinecone with full context vectorization
    """
    BATCH_SIZE = 1  # Reduced due to larger vectors
    upsert_data = []
    total_batches = 0

    try:
        for i, doc in enumerate(documents):
            title = doc["title"]
            content = doc["body"]
            # Combine title and content for vectorization
            full_context = f"Title: {title}\nBody: {content}"

            try:
                # Get embedding for the full context
                emb = get_embedding(full_context)
                if len(emb) != EMBEDDING_DIM:
                    logger.warning(f"Embedding dimension mismatch: got {len(emb)}, expected {EMBEDDING_DIM}")
                    continue

                vector = emb
                upsert_data.append((str(i), vector, {"title": title, "body": content}))

                if len(upsert_data) >= BATCH_SIZE or i == len(documents) - 1:
                    if upsert_data:  # Only upsert if we have data
                        logger.info(f"Upserting batch {total_batches + 1} with {len(upsert_data)} documents")
                        index.upsert(vectors=upsert_data)
                        logger.info(f"Successfully upserted batch {total_batches + 1}")
                        total_batches += 1
                        upsert_data = []

            except Exception as e:
                logger.error(f"Error processing document {i}: {str(e)}")
                if "message length too large" in str(e).lower():
                    BATCH_SIZE = max(1, BATCH_SIZE // 2)
                    logger.warning(f"Reducing batch size to {BATCH_SIZE}")
                    continue

        logger.info(f"Completed upserting {total_batches} batches")
    except Exception as e:
        logger.error(f"Error in upsert_documents: {str(e)}")
        raise


def retrieve_documents(query, top_k=7):
    """
    Retrieves documents from Pinecone based on query embedding
    """
    try:
        # This is the core part that "runs directly from Pinecone"
        if not pinecone_available or index is None:
            logger.warning("Pinecone not available, returning empty results")
            return [], []
            
        query_vector = get_embedding(query)
        if not isinstance(query_vector, list):
            query_vector = query_vector.tolist() # Ensure it's a list for Pinecone query

        query_response = index.query(
            vector=query_vector,
            top_k=top_k,
            include_metadata=True
        )

        results = query_response.get("matches", [])
        retrieved_titles = []
        retrieved_contents = []
        retrieved_scores = []

        for match in results:
            retrieved_titles.append(match["metadata"]["title"])
            retrieved_contents.append(match["metadata"]["body"])
            retrieved_scores.append(match["score"])

        logger.info(f"Retrieved {len(retrieved_titles)} documents for query with scores: {retrieved_scores}")
        return retrieved_titles, retrieved_contents

    except Exception as e:
        logger.error(f"Error retrieving documents: {str(e)}")
        return [], []  # Return empty instead of raising


def remove_thoughts(text):
    """Removes thought process tags from text"""
    return re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL)


async def classify_query(query: str) -> str:
    """
    Classifies if a query needs documents (NS) or can be answered directly (safe)
    """
    try:
        logger.info(f"Classifying query: {query}")

        completion = groq_client.chat.completions.create(
            model="llama-3.1-8b-instant",
            messages=[
                {
                    "role": "system",
                    "content": """You are a dedicated assistant for the news magazine 'Al Mujtama' , AFTER YOU THER IS AN AGENT THAT NEED TO GENERATE THE ANWSERS BASED ON 'Al Mujtama' DATABASE.
                    You represent an AGENT of a process so you are an ai agent that reply with "safe" or "NS" , 
                    if you need more detailles (data/documents) to reply to the users's or visitors query abour 'Al Mujtama' news reply with "NS", 
                    if you do not need more detailles like if the user's is greeting or just asking about you, you must  to reply with "safe", 
                    if you the users is asking about 'Al Mujtama' data like if he wants news or articales or ixplore topics you must reply with "NS" , only with "NS" do not say anything else."""
                },
                {
                    "role": "user",
                    "content": query
                }
            ],
            temperature=0.2,  # Lower temperature for more consistent responses
            max_tokens=10,  # We only need a short response
        )

        response = completion.choices[0].message.content.strip()
        logger.info(f"Query classified as: {response}")

        # Normalize the response to handle potential variations
        if "NS" in response:
            return "NS"
        else:
            return "safe"

    except Exception as e:
        logger.error(f"Error classifying query: {str(e)}")
        # Default to NS (retrieving documents) in case of any errors
        return "NS"


async def safe_generate_answer(query: str) -> dict:
    """
    Generates an answer without retrieving any documents
    """
    try:
        logger.info(f"Processing 'safe' query: {query}")

        prompt = (
            "You are an ARABIC CHATBOT that speaks and understands Arabic fluently using formal Modern Standard Arabic. "
        "You are a dedicated assistant for the news magazine 'Al Mujtama' and must base your responses solely on the Al Mujtama database of articles and news. "
        "When a question relates to current events, news, or articles, ensure your answer is derived exclusively from the verified content available in the Al Mujtama database. "
        f"Question: {query}\n\n"
        "Important Instructions:\n"
        "1. Answer using only the information available in the Al Mujtama database.\n"
        "2. Respond exclusively in Arabic with clear, formal language.\n"
        "3. Provide responses that are concise, thorough, and precise.\n"
        "4. If asked for any information not contained within the Al Mujtama database, politely explain that you can only provide information from this database.\n"

        )

        response = groq_client.chat.completions.create(
            model="meta-llama/llama-4-scout-17b-16e-instruct",
            messages=[
                {"role": "system", "content": """You are an expert Arabic language assistant for the esteemed news magazine 'Al Mujtama.' Your primary role is to provide accurate, insightful, and well-contextualized information about news and articles from the Al Mujtama database. Always respond exclusively in Arabic, using a formal and clear journalistic style that reflects the professionalism and credibility of Al Mujtama.

                                                When users inquire about current events, news articles, or cultural topics, base your responses solely on the verified content available in the Al Mujtama database. Ensure your answers are informative, neutral, and precise. Maintain consistency in tone and provide necessary context or clarifications when needed, while avoiding any personal opinions unless specifically requested.

                                                Your knowledge is focused on the content and archives of Al Mujtama, so always refer to this trusted source for all responses related to news and articles."""},
                {"role": "user", "content": prompt}
            ],
            temperature=0.6,
            max_tokens=1024,
        )

        final_answer = remove_thoughts(response.choices[0].message.content)

        return {
            "response": final_answer,
            "reference": "general knowledge"
        }

    except Exception as e:
        logger.error(f"Error in safe_generate_answer: {str(e)}")
        return {
            "response": "عذراً، حدث خطأ أثناء معالجة طلبك. يرجى المحاولة مرة أخرى.",
            "reference": "",
            "error": str(e)
        }


async def NS_generate_answer(query: str) -> dict:
    """
    Generates an answer with improved context matching
    """
    try:
        logger.info(f"Processing query: {query}")

        retrieved_titles, retrieved_contents = retrieve_documents(query, top_k=7)

        if not retrieved_titles:
            return {
                "response": "عذراً، لم أتمكن من العثور على معلومات ذات صلة في قاعدة البيانات حول استفسارك.",
                "reference": ""
            }

        combined_context = ""
        for title, content in zip(retrieved_titles, retrieved_contents):
            combined_context += f"Document Title: {title}\nDocument Content: {content}\n\n"

        prompt = (
             "You are an ARABIC CHATBOT that speaks and understands Arabic fluently using formal Modern Standard Arabic. "
        "You are a dedicated assistant for the news magazine 'Al Mujtama' 'المجتمع' and must base your responses solely on the Al Mujtama database of articles and news. "
        "When a question relates to current events, news, or articles, ensure your answer is derived exclusively from the verified content available in the Al Mujtama database. "
        f"Question: {query} reply in arabic and only arabic , no other launguges\n\n"
        "Important Instructions:\n"
        "1. Answer using only the information available in the Al Mujtama database.\n"
        "2. Respond exclusively in Arabic with clear, formal language.\n"
        "3. Provide responses that are concise, thorough, and precise.\n"
        "4. If asked for any information not contained within the Al Mujtama database, politely explain that you can only provide information from this database.\n"
        "5. Always respond in Arabic and use only the provided document information.\n"
        "IMPORTANT. VERY IMPORTANT : YOUR REPLAY MUST BE SHORT OR MEDUIM , ONLY IF THE USER ASK'S TO MAKE IT LONG YOU MAKE IT LONG.\n"
        )

        try:
            response = groq_client.chat.completions.create(
                model="meta-llama/llama-4-scout-17b-16e-instruct",
                messages=[
                    {"role": "user", "content": prompt},
                    {"role": "system", "content": (
                        """You only Talk AND REPLY IN ARABIC , You are an expert Arabic language assistant for the esteemed news magazine 'Al Mujtama.' Your primary role is to provide accurate, insightful, and well-contextualized information about news and articles from the Al Mujtama database. Always respond exclusively in Arabic, using a formal and clear journalistic style that reflects the professionalism and credibility of Al Mujtama.

                                                When users inquire about current events, news articles, or cultural topics, base your responses solely on the verified content available in the Al Mujtama database. Ensure your answers are informative, neutral, and precise. Maintain consistency in tone and provide necessary context or clarifications when needed, while avoiding any personal opinions unless specifically requested.

                                                Your knowledge is focused on the content and archives of Al Mujtama, so always refer to this trusted source for all responses related to news and articles.

                                                Always respond in Arabic and use only the provided document information."""
                    )}
                ],
                max_tokens=8000,
                temperature=0.6,
            )

            final_answer = remove_thoughts(response.choices[0].message.content)

            return {
                "response": final_answer,
                "reference": combined_context
            }

        except Exception as e:
            logger.error(f"Error in Together API call: {str(e)}")
            return {
                "response": "عذراً، حدث خطأ أثناء معالجة طلبك. يرجى المحاولة مرة أخرى.",
                "reference": "",
                "error": str(e)
            }

    except Exception as e:
        logger.error(f"Error in generate_answer: {str(e)}")
        return {
            "response": "عذراً، حدث خطأ أثناء معالجة طلبك. يرجى المحاولة مرة أخرى.",
            "reference": "",
            "error": str(e)
        }


class ChatMessage(BaseModel):
    message: str


@app.post("/api/chat")
async def chat(chat_message: ChatMessage):
    try:
        if not chat_message.message.strip():
            raise HTTPException(status_code=400, detail="No message provided")

        logger.info(f"Received chat request: {chat_message.message[:100]}...")

        # Check if Pinecone is available before processing
        if not pinecone_available:
            return JSONResponse(
                status_code=200, # Using 200 with an error message in body for user-friendly display
                content={
                    "response": "عذراً، الخدمة غير متاحة حالياً. يرجى المحاولة مرة أخرى لاحقاً.",
                    "reference": "",
                    "error": "Pinecone service unavailable"
                }
            )

        # First, classify the query
        try:
            query_type = await classify_query(chat_message.message)
            logger.info(f"Query classified as: {query_type}")
        except Exception as e:
            logger.error(f"Error in query classification: {str(e)}")
            return JSONResponse(
                status_code=200,
                content={
                    "response": "عذراً، حدث خطأ في تصنيف الاستفسار. يرجى المحاولة مرة أخرى.",
                    "reference": "",
                    "error": f"Classification error: {str(e)}"
                }
            )

        # Process based on classification
        try:
            if query_type == "safe":
                answer_data = await safe_generate_answer(chat_message.message)
            else:  # "NS"
                answer_data = await NS_generate_answer(chat_message.message)
        except Exception as e:
            logger.error(f"Error in answer generation: {str(e)}")
            return JSONResponse(
                status_code=200,
                content={
                    "response": "عذراً، حدث خطأ أثناء إنتاج الإجابة. يرجى المحاولة مرة أخرى.",
                    "reference": "",
                    "error": f"Answer generation error: {str(e)}"
                }
            )

        if "error" in answer_data:
            logger.error(f"Error in processing: {answer_data['error']}")
            return JSONResponse(
                status_code=200,
                content=answer_data
            )

        # Add query type to response for debugging/analytics
        answer_data["query_type"] = query_type
        return answer_data

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Unexpected error in chat endpoint: {str(e)}")
        return JSONResponse(
            status_code=200,
            content={
                "response": "عذراً، حدث خطأ غير متوقع. يرجى المحاولة مرة أخرى.",
                "reference": "",
                "error": f"Unexpected error: {str(e)}"
            }
        )


@app.post("/api/reindex")
async def reindex():
    """
    Endpoint to force reindexing of all documents.
    This will delete all existing vectors and re-upsert.
    """
    try:
        if not pinecone_available or index is None:
            raise HTTPException(status_code=503, detail="Pinecone service unavailable")

        logger.info("Starting reindexing process: Deleting all existing vectors...")
        index.delete(delete_all=True)
        logger.info("Existing vectors deleted. Reinserting documents...")
        # Reinsert with new vectorization
        upsert_documents()
        
        # Get updated stats
        stats = index.describe_index_stats()
        vector_count = stats.get('total_vector_count', 0)
        logger.info(f"Reindexing completed. Total vectors in index: {vector_count}")

        return {"status": "success", "message": f"Reindexing completed. Total vectors: {vector_count}"}
    except Exception as e:
        logger.error(f"Error during reindexing: {str(e)}")
        return {"status": "error", "message": str(e)}


@app.get("/", response_class=HTMLResponse)
async def read_root(request: Request):
    return templates.TemplateResponse("cleT.html", {"request": request})

@app.get("/chat", response_class=HTMLResponse)
async def read_root(request: Request):
    return templates.TemplateResponse("cleT.html", {"request": request})


@app.get("/api/documents")
async def get_documents():
    # return documents
    return {"message": "Direct document serving is disabled. Data is accessed via Pinecone."}


@app.get("/database", response_class=HTMLResponse)
async def read_database(request: Request):
    return templates.TemplateResponse("Datasets.html", {"request": request})


@app.get("/debug")
async def debug_status():
    """
    Endpoint for debugging system status
    """
    try:
        # Check Pinecone vector count
        pinecone_vectors = 0
        pinecone_status = "unavailable"
        
        if pinecone_available and index is not None:
            try:
                stats = index.describe_index_stats()
                pinecone_vectors = stats.get('total_vector_count', 0)
                pinecone_status = "connected"
            except Exception as e:
                logger.error(f"Error getting Pinecone stats: {str(e)}")
                pinecone_status = f"error: {str(e)}"
        
        return {
            "status": "running",
            "documents_loaded": len(documents), # This is local memory, not Pinecone
            "pinecone_available": pinecone_available,
            "pinecone_status": pinecone_status,
            "pinecone_vectors_in_index": pinecone_vectors,
            "message": "Debug endpoint working"
        }
    except Exception as e:
        return {"status": "error", "error": str(e)}


@app.get("/health")
async def health_check():
    """Simple health check endpoint"""
    return {"status": "healthy", "message": "FastAPI is running on Vercel"}

@app.get("/api/test")
async def test_endpoint():
    """Test endpoint to verify basic functionality"""
    return {"message": "API is working", "endpoints": ["health", "api/test", "api/chat", "debug"]}


@app.post("/api/seed-pinecone")
async def seed_pinecone():
    """
    Endpoint to manually seed Pinecone with documents (for demo purposes or initial population).
    This will upsert documents without deleting existing ones.
    """
    try:
        if not pinecone_available:
            raise HTTPException(status_code=503, detail="Pinecone not available")
        
        if len(documents) == 0:
            return {"status": "error", "message": "No local documents loaded to seed"}
        
        logger.info(f"Starting to seed Pinecone with {len(documents)} documents (this will add/update, not delete)")
        upsert_documents()
        
        # Get updated stats
        stats = index.describe_index_stats()
        vector_count = stats.get('total_vector_count', 0)
        
        return {
            "status": "success", 
            "message": f"Successfully seeded Pinecone with {len(documents)} documents",
            "total_vectors_in_index": vector_count
        }
    except Exception as e:
        logger.error(f"Error seeding Pinecone: {str(e)}")
        return {"status": "error", "message": str(e)}


if __name__ == "__main__":
    import uvicorn

    # Run the app with uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)