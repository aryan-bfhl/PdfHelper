import os
import io
import requests
import PyPDF2
from google import genai 
from pinecone import Pinecone, ServerlessSpec 
from groq import Groq
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from typing import List, Dict, Tuple
import logging
import uuid
import time
from dotenv import load_dotenv
import asyncio

load_dotenv()

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# --- Configuration ---
GEMINI_API_KEY = os.getenv("GEMINI_KEY")
PINECONE_API_KEY = os.getenv("PINECONE_KEY")
PINECONE_ENVIRONMENT = os.getenv("PINECONE_ENV") 
GROQ_API_KEY = os.getenv("GROQ_KEY")

PINECONE_INDEX_NAME = "new" 
EMBEDDING_DIMENSION = 3072 

GEMINI_LLM_MODEL = "gemini-1.5-flash"
GEMINI_EMBEDDING_MODEL = "models/gemini-embedding-001" 
GROQ_LLM_MODEL = "llama3-8b-8192"

CHUNK_SIZE = 1000
CHUNK_OVERLAP = 100

# GEMINI EMBEDDING API BATCH LIMIT
MAX_EMBEDDING_BATCH_SIZE = 100 # Maximum number of texts per batch request

# --- FastAPI App Initialization ---
app = FastAPI(
    title="PDF RAG API",
    description="API to extract text from PDF, generate embeddings, and answer questions using LLM with RAG.",
    version="1.0.0"
)

# --- Initialize External Clients ---
genai_client = None 
if GEMINI_API_KEY:
    try:
        genai_client = genai.Client(api_key=GEMINI_API_KEY) 
        logger.info("Gemini API client initialized.")
    except Exception as e:
        logger.error(f"Failed to initialize Gemini API client: {e}. Gemini services will not be available.", exc_info=True)
else:
    logger.error("GEMINI_KEY environment variable not set. Gemini services will not be available.")

pinecone_client = None
pinecone_index = None

if PINECONE_API_KEY and PINECONE_ENVIRONMENT:
    try:
        pinecone_client = Pinecone(api_key=PINECONE_API_KEY, environment=PINECONE_ENVIRONMENT)
        logger.info("Pinecone client initialized.")
        pinecone_index = pinecone_client.Index(PINECONE_INDEX_NAME) 
        logger.info(f"Connected to Pinecone index: {PINECONE_INDEX_NAME}")

    except Exception as e:
        logger.error(f"Failed to initialize Pinecone. Please check PINECONE_KEY, PINECONE_ENV, or index configuration: {e}", exc_info=True)
else:
    logger.error("PINECONE_KEY or PINECONE_ENV not set. Pinecone services will not be available.")

groq_client = None
if GROQ_API_KEY:
    try:
        groq_client = Groq(api_key=GROQ_API_KEY)
        logger.info("Groq client initialized.")
    except Exception as e:
        logger.warning(f"Failed to initialize Groq client: {e}. Groq LLM will not be available.")

# --- Pydantic Models for API Request/Response ---
class HackRXRequest(BaseModel):
    documents: str = Field(..., description="URL to the PDF document.")
    questions: List[str] = Field(..., description="List of questions to ask about the PDF.")

class HackRXResponse(BaseModel):
    answers: List[str] = Field(..., description="List of answers corresponding to the questions.")

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
        raise HTTPException(status_code=400, detail=f"Invalid or corrupted PDF content: {e}")
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

async def generate_embeddings(texts: List[str]) -> Tuple[List[str], List[List[float]]]:
    """Generates embeddings and returns both embeddings and filtered input texts."""
    if genai_client is None: 
        raise HTTPException(status_code=500, detail="Gemini API client is not initialized.")

    all_embeddings = []
    successful_texts = []

    filtered_texts = [text for text in texts if text.strip()]
    if not filtered_texts:
        logger.warning("No valid text chunks after filtering.")
        return [], []

    for i in range(0, len(filtered_texts), MAX_EMBEDDING_BATCH_SIZE):
        batch_texts = filtered_texts[i:i + MAX_EMBEDDING_BATCH_SIZE]
        try:
            response = genai_client.models.embed_content(
                model=GEMINI_EMBEDDING_MODEL,
                contents=batch_texts,
            )

            if not hasattr(response, 'embeddings') or not isinstance(response.embeddings, list):
                logger.error(f"No embeddings returned for batch starting at index {i}. Skipping batch.")
                continue

            for j, embedding_obj in enumerate(response.embeddings):
                if hasattr(embedding_obj, 'value') and isinstance(embedding_obj.value, list):
                    all_embeddings.append(embedding_obj.value)
                    successful_texts.append(batch_texts[j])
                else:
                    logger.warning(f"Invalid embedding for chunk index {i + j}. Skipping.")

        except Exception as e:
            logger.error(f"Embedding failed for batch {i}: {e}", exc_info=True)
            continue

    logger.info(f"Generated {len(all_embeddings)} embeddings from {len(texts)} input chunks.")
    return successful_texts, all_embeddings

async def upsert_embeddings_to_pinecone(index, text_chunks: List[str], embeddings: List[List[float]], document_id: str):
    """Upserts embeddings and their corresponding text chunks (metadata) to Pinecone."""
    if not index:
        raise HTTPException(status_code=500, detail="Pinecone index is not initialized or usable.")
    
    if len(text_chunks) != len(embeddings):
        logger.error(f"Mismatched lengths of text_chunks ({len(text_chunks)}) and embeddings ({len(embeddings)}) for upsert. This indicates an issue upstream.")
        raise HTTPException(status_code=500, detail="Internal error: Mismatch between chunks and embeddings. Data integrity issue.")

    vectors_to_upsert = []
    for i in range(len(embeddings)): # Iterate based on the length of embeddings, as these are the ones actually generated
        vector_id = f"{document_id}-chunk-{i}"
        vectors_to_upsert.append({
            "id": vector_id,
            "values": embeddings[i],
            "metadata": {"text_chunk": text_chunks[i], "document_id": document_id, "chunk_index": i}
        })
    
    if not vectors_to_upsert:
        logger.warning("No vectors to upsert to Pinecone. This might mean no text was extracted or no embeddings were generated.")
        return 

    try:
        upsert_batch_size = 100 # Pinecone also has batch limits, 100 is a good default
        for i in range(0, len(vectors_to_upsert), upsert_batch_size):
            batch = vectors_to_upsert[i:i + upsert_batch_size]
            index.upsert(vectors=batch)
            logger.info(f"Upserted batch {i//upsert_batch_size + 1}/{(len(vectors_to_upsert) + upsert_batch_size - 1)//upsert_batch_size} to Pinecone.")
            await asyncio.sleep(0.05) # Small delay between Pinecone upsert batches
        logger.info(f"Successfully upserted {len(vectors_to_upsert)} embeddings to Pinecone for document {document_id}.")
    except Exception as e:
        logger.error(f"Error upserting embeddings to Pinecone: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Failed to upsert embeddings to Pinecone: {e}")

async def query_pinecone_and_rag(index, question: str, llm_client_type: str = "gemini", top_k: int = 3) -> str:
    """
    Queries Pinecone for relevant context based on the question,
    then uses an LLM (Gemini or Groq) to answer the question with RAG.
    `llm_client_type` can be "gemini" or "groq".
    """
    if not index:
        raise HTTPException(status_code=500, detail="Pinecone index is not initialized or usable.")

    if llm_client_type == "gemini" and genai_client is None: 
        raise HTTPException(status_code=500, detail="Gemini API client not initialized for Gemini LLM. Cannot proceed with Gemini.")
    elif llm_client_type == "groq" and groq_client is None:
        raise HTTPException(status_code=500, detail="Groq client not initialized for Groq LLM. Cannot proceed with Groq.")

    try:
        # 1. Generate embedding for the user's question using the new client
        try:
            query_embedding_response = await genai_client.models.embed_content( 
                model=GEMINI_EMBEDDING_MODEL,
                contents=[question], # Pass as a list, even for a single query
                task_type="RETRIEVAL_QUERY", # Keep as RETRIEVAL_QUERY for RAG
            )
        except Exception as e:
             logger.error(f"Error calling Gemini embed_content API for query: {e}", exc_info=True)
             raise ValueError(f"Gemini API query call failed: {e}")

        if not hasattr(query_embedding_response, 'embeddings') or not isinstance(query_embedding_response.embeddings, list) or not query_embedding_response.embeddings:
            error_message = "Gemini API response for query embedding missing 'embeddings' attribute or it's empty."
            if hasattr(query_embedding_response, 'error') and query_embedding_response.error:
                error_details = f"Error object: {query_embedding_response.error}"
                if hasattr(query_embedding_response.error, 'message'):
                    error_details += f" | Message: {query_embedding_response.error.message}"
                if hasattr(query_embedding_response.error, 'code'):
                    error_details += f" | Code: {query_embedding_response.error.code}"
                error_message = f"Gemini API returned an error for query embedding: {error_details}"
            logger.error(error_message)
            raise ValueError(error_message)

        query_embedding = query_embedding_response.embeddings[0].value 

        # 2. Query Pinecone to find top_k most relevant text chunks
        query_results = index.query(
            vector=query_embedding,
            top_k=top_k,
            include_metadata=True 
        )
        
        context_chunks = [match.metadata['text_chunk'] for match in query_results.matches if 'text_chunk' in match.metadata]
        
        if not context_chunks:
            logger.warning(f"No relevant context found in Pinecone for question: '{question}'. Answering without specific context.")
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
            model = genai_client.GenerativeModel(GEMINI_LLM_MODEL)
            try:
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
    if genai_client is None:
        raise HTTPException(status_code=500, detail="Gemini API client is not initialized. Cannot proceed with Gemini services.")
    if pinecone_index is None: 
        raise HTTPException(status_code=500, detail="Pinecone client or index failed to initialize or is configured incorrectly. Check your Pinecone API key, environment, and index dimension.")

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

        if len(filtered_text_chunks_for_embedding) != len(embeddings):
            logger.error(f"Critical Mismatch: The number of text chunks sent for embedding ({len(filtered_text_chunks_for_embedding)}) does not match the number of generated embeddings ({len(embeddings)}). This indicates an API issue or data integrity problem.")
            raise HTTPException(status_code=500, detail="Internal data processing error: Mismatch in chunk-embedding count from API.")

        document_id = str(uuid.uuid4()) 
        
        await upsert_embeddings_to_pinecone(pinecone_index, filtered_text_chunks_for_embedding, embeddings, document_id)
        
        answers = []
        for question in request.questions:
            answer = await query_pinecone_and_rag(pinecone_index, question, llm_client_type="gemini") 
            answers.append(answer)
        
        logger.info("Successfully processed PDF and answered all questions.")
        return HackRXResponse(answers=answers)

    except HTTPException as e:
        logger.error(f"API Error: {e.detail}", exc_info=True)
        raise e
    except Exception as e:
        logger.error(f"An unhandled error occurred during PDF processing: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"An internal server error occurred: {e}")