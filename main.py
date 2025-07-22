import os
import io
import requests
import PyPDF2
import google.generativeai as genai
from google.generativeai import GenerativeModel
from groq import Groq
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from typing import List, Dict, Tuple
import logging
import uuid
import time
import asyncio
from dotenv import load_dotenv

# --- MongoDB Specific Imports ---
from pymongo import MongoClient
from pymongo.errors import ConnectionFailure, OperationFailure
import certifi
ca = certifi.where()

# --- LangChain Specific Imports ---
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_core.embeddings import Embeddings

load_dotenv()
os.environ["GOOGLE_API_KEY"] = os.getenv("GEMINI_KEY", "")

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# --- Configuration ---
GEMINI_API_KEY = os.getenv("GEMINI_KEY")
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

# Initialize Gemini SDK
if GEMINI_API_KEY:
    try:
        genai.configure(api_key=GEMINI_API_KEY)
        logger.info("Gemini SDK configured successfully.")
    except Exception as e:
        logger.error(f"Failed to configure Gemini SDK: {e}. Gemini services will not be available.", exc_info=True)
        genai_client = None
else:
    logger.error("GEMINI_KEY environment variable not set. Gemini services will not be available.")
    genai_client = None

# --- MongoDB Configuration ---
MONGO_URI = os.getenv("MONGO_URI")
MONGO_DB_NAME = "rag_db"
MONGO_COLLECTION_NAME = "pdf_chunks"
MONGO_VECTOR_INDEX_NAME = "default"

EMBEDDING_DIMENSION = 3072

GEMINI_LLM_MODEL = "gemini-1.5-flash"
GEMINI_EMBEDDING_MODEL = "gemini-embedding-001"
GROQ_LLM_MODEL = "llama3-8b-8192"

CHUNK_SIZE = 1000
CHUNK_OVERLAP = 100

# --- FastAPI App Initialization ---
app = FastAPI(
    title="PDF RAG API with MongoDB",
    description="API to extract text from PDF, generate embeddings, and answer questions using LLM with RAG.",
    version="1.0.0"
)

# --- Initialize External Clients ---
genai_client = None  # Not needed, as we use genai.GenerativeModel directly
embeddings_model: Embeddings = None
if GEMINI_API_KEY:
    try:
        embeddings_model = GoogleGenerativeAIEmbeddings(
            model=GEMINI_EMBEDDING_MODEL,
            google_api_key=GEMINI_API_KEY,
        )
        logger.info(f"LangChain GoogleGenerativeAIEmbeddings initialized with model: {GEMINI_EMBEDDING_MODEL}")
    except Exception as e:
        logger.error(f"Failed to initialize LangChain GoogleGenerativeAIEmbeddings: {e}. Embedding services will not be available.", exc_info=True)

mongo_client = None
mongo_collection = None

if MONGO_URI:
    try:
        mongo_client = MongoClient(MONGO_URI, tlsCAFile=ca)
        mongo_client.admin.command('ping')
        mongo_collection = mongo_client[MONGO_DB_NAME][MONGO_COLLECTION_NAME]
        logger.info("MongoDB client and collection initialized.")
        logger.info(f"Connected to MongoDB database: '{MONGO_DB_NAME}', collection: '{MONGO_COLLECTION_NAME}'")
    except ConnectionFailure as e:
        logger.error(f"Failed to connect to MongoDB Atlas. Check MONGO_URI and network access: {e}", exc_info=True)
        mongo_client = None
        mongo_collection = None
    except Exception as e:
        logger.error(f"An unexpected error occurred during MongoDB initialization: {e}", exc_info=True)
        mongo_client = None
        mongo_collection = None
else:
    logger.error("MONGO_URI environment variable not set. MongoDB services will not be available.")

groq_client = None
if GROQ_API_KEY:
    try:
        groq_client = Groq(api_key=GROQ_API_KEY)
        logger.info("Groq client initialized.")
    except Exception as e:
        logger.warning(f"Failed to initialize Groq client: {e}. Groq LLM will not be available.")

# --- Pydantic Models ---
class HackRXRequest(BaseModel):
    documents: str = Field(..., description="URL of the PDF document.")
    questions: List[str] = Field(..., description="List of questions to answer.")

class HackRXResponse(BaseModel):
    answers: List[str]

# --- Core Helper Functions ---
async def download_pdf(url: str) -> bytes:
    """Downloads a PDF from a given URL and returns its content as bytes."""
    try:
        response = requests.get(url, stream=True)
        response.raise_for_status()
        pdf_content = response.content
        logger.info(f"Successfully downloaded PDF from {url}")
        return pdf_content
    except requests.exceptions.RequestException as e:
        logger.error(f"Error downloading PDF from {url}: {e}")
        raise HTTPException(status_code=400, detail=f"Failed to download PDF: {e}. Please check the URL.")

async def extract_text_from_pdf(pdf_content: bytes) -> str:
    """
    Extracts text from a PDF document using PyPDF2.
    """
    text = ""
    try:
        pdf_file = io.BytesIO(pdf_content)
        reader = PyPDF2.PdfReader(pdf_file)

        if not reader.pages:
            raise HTTPException(status_code=400, detail="The PDF document appears to be empty or corrupted.")

        for page_num in range(len(reader.pages)):
            page = reader.pages[page_num]
            extracted_page_text = page.extract_text()
            if extracted_page_text:
                text += extracted_page_text + "\n"
            else:
                logger.warning(f"No text extracted from page {page_num + 1}. This page might contain only images or be malformed.")

        if not text.strip():
            logger.warning("No meaningful text extracted using PyPDF2 from the entire PDF. This might indicate an issue with the PDF content or format.")
            raise HTTPException(status_code=400, detail="Could not extract any text from the PDF. It might be empty or in a format not supported for text extraction.")

        logger.info(f"Extracted total characters from PDF: {len(text)}")
        return text

    except PyPDF2.errors.PdfReadError as e:
        logger.error(f"Error reading PDF content (PyPDF2): {e}")
        raise HTTPException(status_code=400, detail=f"Invalid or corrupted PDF content:{e}")
    except Exception as e:
        logger.error(f"An unexpected error occurred during PDF text extraction: {e}")
        raise HTTPException(status_code=500, detail=f"Internal server error during PDF processing: {e}")

def chunk_text(text: str, chunk_size: int, chunk_overlap: int) -> List[str]:
    """
    Splits a long text into smaller, overlapping chunks.
    This helps in maintaining context for embeddings and RAG.
    """
    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunk = text[start:min(end, len(text))]
        chunks.append(chunk)
        if end >= len(text):
            break
        start += chunk_size - chunk_overlap
    logger.info(f"Text chunked into {len(chunks)} pieces.")
    return chunks

async def generate_embeddings(texts: List[str]) -> List[List[float]]:
    """Generates embeddings for a list of text chunks using LangChain's GoogleGenerativeAIEmbeddings."""
    if embeddings_model is None:
        raise HTTPException(status_code=500, detail="LangChain GoogleGenerativeAIEmbeddings is not initialized. Cannot generate embeddings.")

    filtered_texts = [text for text in texts if text and text.strip()]
    if not filtered_texts:
        logger.warning("No valid text chunks to generate embeddings for after filtering empty/whitespace.")
        return []

    logger.info(f"Attempting to generate embeddings for {len(filtered_texts)} total chunks using LangChain.")

    try:
        embeddings = await embeddings_model.aembed_documents(
            filtered_texts,
            task_type="RETRIEVAL_DOCUMENT"
        )
        logger.info(f"Successfully generated {len(embeddings)} embeddings in total using LangChain.")
        if embeddings and len(embeddings[0]) != EMBEDDING_DIMENSION:
            logger.error(f"Generated embedding dimension ({len(embeddings[0])}) does not match expected dimension ({EMBEDDING_DIMENSION}).")
            raise HTTPException(status_code=500, detail=f"Embedding dimension mismatch: Expected {EMBEDDING_DIMENSION}, got {len(embeddings[0])}.")

        return embeddings
    except Exception as e:
        logger.error(f"LangChain GoogleGenerativeAIEmbeddings failed during embedding generation: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Failed to generate embeddings: {e}")

async def store_embeddings_in_mongodb(collection, text_chunks: List[str], embeddings: List[List[float]], document_id: str):
    """Stores embeddings and their corresponding text chunks (metadata) in MongoDB."""
    if collection is None:
        raise HTTPException(status_code=500, detail="MongoDB collection is not initialized or usable.")

    if len(text_chunks) != len(embeddings):
        logger.error(f"Mismatched lengths of text_chunks ({len(text_chunks)}) and embeddings ({len(embeddings)}) for storage.")
        raise HTTPException(status_code=500, detail="Internal error: Mismatch between chunks and embeddings. Data integrity issue.")

    documents_to_insert = []
    for i in range(len(embeddings)):
        doc = {
            "text_chunk": text_chunks[i],
            "embedding": embeddings[i],
            "document_id": document_id,
            "chunk_index": i,
            "timestamp": time.time()
        }
        documents_to_insert.append(doc)

    if not documents_to_insert:
        logger.warning("No documents to insert into MongoDB. This might mean no text was extracted or no embeddings were generated.")
        return

    try:
        result = collection.insert_many(documents_to_insert)
        logger.info(f"Successfully inserted {len(result.inserted_ids)} documents into MongoDB for document {document_id}.")
    except OperationFailure as e:
        logger.error(f"MongoDB OperationFailure during insertion: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Failed to insert documents into MongoDB: {e.details.get('errmsg', 'Unknown MongoDB error')}")
    except Exception as e:
        logger.error(f"Error inserting embeddings into MongoDB: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Failed to store embeddings: {e}")

async def query_mongodb_and_rag(collection, question: str, llm_client_type: str = "gemini", top_k: int = 3) -> str:
    """
    Queries MongoDB for relevant context based on the question,
    then uses an LLM (Gemini or Groq) to answer the question with RAG.
    `llm_client_type` can be "gemini" or "groq".
    """
    if collection is None:
        raise HTTPException(status_code=500, detail="MongoDB collection is not initialized or usable.")
    if embeddings_model is None:
        raise HTTPException(status_code=500, detail="LangChain GoogleGenerativeAIEmbeddings is not initialized. Cannot generate query embedding.")
    if llm_client_type == "gemini" and not GEMINI_API_KEY:
        raise HTTPException(status_code=500, detail="GEMINI_KEY is not set for Gemini LLM.")
    elif llm_client_type == "groq" and groq_client is None:
        raise HTTPException(status_code=500, detail="Groq client not initialized for Groq LLM. Cannot proceed with Groq.")

    try:
        # 1. Generate embedding for the user's question using LangChain
        try:
            query_embedding = await embeddings_model.aembed_query(
                question,
                task_type="RETRIEVAL_QUERY"
            )
        except Exception as e:
            logger.error(f"Error calling LangChain embed_query for query: {e}", exc_info=True)
            raise ValueError(f"LangChain GoogleGenerativeAIEmbeddings query call failed: {e}")

        # 2. Query MongoDB using Atlas Vector Search aggregation pipeline
        atlas_vector_search_pipeline = [
            {
                "$vectorSearch": {
                    "queryVector": query_embedding,
                    "path": "embedding",
                    "numCandidates": top_k * 40,
                    "limit": top_k,
                    "index": MONGO_VECTOR_INDEX_NAME
                }
            },
            {
                "$project": {
                    "text_chunk": 1,
                    "_id": 0,
                    "score": { "$meta": "vectorSearchScore" }
                }
            }
        ]

        # logger.info(f"vector search:{atlas_vector_search_pipeline}")

        query_results = list(collection.aggregate(atlas_vector_search_pipeline))

        logger.info(query_results)

        context_chunks = [result['text_chunk'] for result in query_results if 'text_chunk' in result]

        if not context_chunks:
            logger.warning(f"No relevant context found in MongoDB for question: '{question}'. Answering without specific context.")
            context_string = "No specific context found in the document. Please note: The answer might be general without document-specific context."
        else:
            context_string = "\n\n".join(context_chunks)
            logger.info(f"Retrieved {len(context_chunks)} context chunks for question: '{question}'")

        # 3. Construct the prompt for the LLM with the retrieved context
        prompt = f"""
        You are an AI assistant. Use the following context to answer the question.
        If the answer is not explicitly available in the provided context, state that you don't know or that the information is not in the document.

        Context:
        {context_string}

        Question: {question}

        Answer:
        """

        # 4. Call the chosen LLM (Gemini or Groq)
        if llm_client_type == "gemini":
            if not GEMINI_API_KEY:
                raise HTTPException(status_code=500, detail="GEMINI_KEY is not set for Gemini LLM.")
            try:
                model = genai.GenerativeModel(GEMINI_LLM_MODEL)
                response = await model.generate_content_async(prompt)
                answer = response.text
                logger.info(f"Gemini LLM answered question: '{question}'")
            except Exception as e:
                logger.error(f"Error calling Gemini GenerativeModel API: {e}", exc_info=True)
                raise ValueError(f"Gemini LLM call failed: {e}")
        elif llm_client_type == "groq":
            if not groq_client:
                raise HTTPException(status_code=500, detail="Groq client not initialized. Cannot use Groq LLM.")
            try:
                chat_completion = groq_client.chat.completions.create(
                    messages=[
                        {
                            "role": "user",
                            "content": prompt,
                        }
                    ],
                    model=GROQ_LLM_MODEL,
                    temperature=0.7,
                    max_tokens=500,
                )
                answer = chat_completion.choices[0].message.content
                logger.info(f"Groq LLM answered question: '{question}'")
            except Exception as e:
                logger.error(f"Error calling Groq API: {e}", exc_info=True)
                raise ValueError(f"Groq API call failed: {e}")
        else:
            raise ValueError("Invalid LLM client type specified. Must be 'gemini' or 'groq'.")

        return answer

    except ValueError as ve:
        logger.error(f"Error from LLM API during RAG process for question '{question}': {ve}")
        raise HTTPException(status_code=500, detail=f"Failed to answer question (LLM API error): {ve}")
    except Exception as e:
        logger.error(f"An unexpected error occurred during RAG process for question '{question}': {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Failed to answer question due to an internal error: {e}")

# --- FastAPI Endpoint ---
@app.post("/hackrx/run", response_model=HackRXResponse)
async def hackrx_run_endpoint(request: HackRXRequest):
    """
    API endpoint to process a PDF from a blob link, generate embeddings,
    and answer a list of questions using RAG, matching HackRX specifications.
    """
    if not GEMINI_API_KEY and groq_client is None:
        raise HTTPException(status_code=500, detail="Neither Gemini nor Groq LLM client is initialized. Cannot answer questions.")
    if mongo_collection is None:
        raise HTTPException(status_code=500, detail="MongoDB client or collection failed to initialize. Check your MONGO_URI and Atlas setup.")
    if not GEMINI_API_KEY:
        raise HTTPException(status_code=500, detail="Gemini API key is not set. Cannot generate embeddings.")

    logger.info(f"Received request to process PDF from: {request.documents}")
    logger.info(f"Questions received: {request.questions}")

    try:
        pdf_content = await download_pdf(request.documents)
        extracted_text = await extract_text_from_pdf(pdf_content)
        text_chunks = chunk_text(extracted_text, CHUNK_SIZE, CHUNK_OVERLAP)

        filtered_text_chunks_for_embedding = [chunk for chunk in text_chunks if chunk and chunk.strip()]

        if not filtered_text_chunks_for_embedding:
            raise HTTPException(status_code=400, detail="No usable text chunks extracted from the PDF after filtering.")

        embeddings = await generate_embeddings(filtered_text_chunks_for_embedding)

        if embeddings and len(embeddings[0]) != EMBEDDING_DIMENSION:
            raise HTTPException(status_code=500, detail=f"Generated embedding dimension ({len(embeddings[0])}) does not match expected dimension ({EMBEDDING_DIMENSION}). This indicates an issue with the chosen embedding model configuration.")

        if len(filtered_text_chunks_for_embedding) != len(embeddings):
            logger.error(f"Critical Mismatch: The number of text chunks sent for embedding ({len(filtered_text_chunks_for_embedding)}) does not match the number of generated embeddings ({len(embeddings)}). This indicates an API issue or data integrity problem.")
            raise HTTPException(status_code=500, detail="Internal data processing error: Mismatch in chunk-embedding count from API.")

        document_id = str(uuid.uuid4())

        await store_embeddings_in_mongodb(mongo_collection, filtered_text_chunks_for_embedding, embeddings, document_id)

        answers = []
        for question in request.questions:
            answer = await query_mongodb_and_rag(mongo_collection, question, llm_client_type="gemini")
            answers.append(answer)

        logger.info("Successfully processed PDF and answered all questions.")
        logger.info(answers)
        return HackRXResponse(answers=answers)

    except HTTPException as e:
        logger.error(f"API Error: {e.detail}", exc_info=True)
        raise e
    except Exception as e:
        logger.error(f"An unhandled error occurred during PDF processing: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"An internal server error occurred: {e}")