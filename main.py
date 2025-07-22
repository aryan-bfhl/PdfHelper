import os
import io
import requests
import PyPDF2
import google.generativeai as genai
from pinecone import Pinecone
from groq import Groq
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from typing import List, Dict, Tuple
import logging
import uuid
from dotenv import load_dotenv

load_dotenv()  # Loads from .env file into os.environ

# Configure logging for better visibility
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# --- Configuration ---
GEMINI_API_KEY = os.getenv("GEMINI_KEY")
PINECONE_API_KEY = os.getenv("PINECONE_KEY")
PINECONE_ENVIRONMENT = os.getenv("PINECONE_ENV") # e.g., 'us-west-2' or 'gcp-starter'
GROQ_API_KEY = os.getenv("GROQ_KEY")

# Pinecone index configuration
PINECONE_INDEX_NAME = "new" # Ensure this index name exists in your Pinecone project
EMBEDDING_DIMENSION = 768 # This is the dimension for Gemini's 'text-embedding-004' model

# LLM model choices
GEMINI_LLM_MODEL = "gemini-1.5-flash"
GEMINI_EMBEDDING_MODEL = "text-embedding-004"
GROQ_LLM_MODEL = "llama3-8b-8192" # Example Groq model, check Groq documentation for available models

# Text chunking parameters for RAG
CHUNK_SIZE = 1000  # Max characters per text chunk
CHUNK_OVERLAP = 100 # Overlap between chunks to maintain context

# --- FastAPI App Initialization ---
app = FastAPI(
    title="PDF RAG API",
    description="API to extract text from PDF, generate embeddings, and answer questions using LLM with RAG.",
    version="1.0.0"
)

# --- Initialize External Clients ---
# Initialize Gemini Generative AI client
if GEMINI_API_KEY:
    genai.configure(api_key=GEMINI_API_KEY)
    logger.info("Gemini API configured.")
else:
    logger.error("GEMINI_KEY environment variable not set. Gemini services will not be available.")

# Initialize Pinecone client and index
pinecone_client = None
pinecone_index = None # This will hold the pinecone_client.Index object

if PINECONE_API_KEY and PINECONE_ENVIRONMENT: # Ensure both are set for Pinecone initialization
    try:
        # Correct Pinecone client initialization - Pass both API key and environment
        pinecone_client = Pinecone(api_key=PINECONE_API_KEY, environment=PINECONE_ENVIRONMENT)
        logger.info("Pinecone client initialized.")
        
        # Check if the Pinecone index exists, create it if not
        existing_indexes = pinecone_client.list_indexes()
        existing_index_names = [idx['name'] for idx in existing_indexes]

        if PINECONE_INDEX_NAME not in existing_index_names:
            logger.info(f"Creating Pinecone index: {PINECONE_INDEX_NAME} with dimension {EMBEDDING_DIMENSION}")
            pinecone_client.create_index(
                name=PINECONE_INDEX_NAME,
                dimension=EMBEDDING_DIMENSION,
                metric='cosine', # Cosine similarity is common for embeddings
                spec={"serverless": {"cloud": "aws", "region": "us-west-2"}} # Example for AWS serverless
                # For pod-based, use: spec={"pod": {"environment": PINECONE_ENVIRONMENT, "pod_type": "p1.x1", "pods": 1}}
            )
        
        # Get the Index object after ensuring it exists
        pinecone_index = pinecone_client.Index(PINECONE_INDEX_NAME) 
        logger.info(f"Connected to Pinecone index: {PINECONE_INDEX_NAME}")
    except Exception as e:
        logger.error(f"Failed to initialize Pinecone. Please check PINECONE_KEY and PINECONE_ENV: {e}", exc_info=True)
else:
    logger.error("PINECONE_KEY or PINECONE_ENV not set. Pinecone services will not be available.")

# Initialize Groq client (optional)
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
        response.raise_for_status()  # Raise an HTTPError for bad responses (4xx or 5xx)
        pdf_content = response.content
        logger.info(f"Successfully downloaded PDF from {url}")
        return pdf_content
    except requests.exceptions.RequestException as e:
        logger.error(f"Error downloading PDF from {url}: {e}")
        raise HTTPException(status_code=400, detail=f"Failed to download PDF: {e}. Please check the URL.")

async def extract_text_from_pdf(pdf_content: bytes) -> str:
    """
    Extracts text from a PDF document using PyPDF2.
    This function is designed for text-based PDFs.
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

async def generate_embeddings(texts: List[str]) -> List[List[float]]:
    """Generates embeddings for a list of text chunks using Gemini's embedding model."""
    if not GEMINI_API_KEY:
        raise HTTPException(status_code=500, detail="Gemini API key not configured.")
    
    embeddings = []
    try:
        batch_size = 100 
        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i:i + batch_size]
            
            if not batch_texts: 
                continue

            response = genai.embed_content(
                model=GEMINI_EMBEDDING_MODEL,
                content=batch_texts,
                task_type="RETRIEVAL_DOCUMENT"
            )
            
            # --- CRITICAL CHANGE HERE: Accessing embeddings attribute ---
            # The 'response' object from genai.embed_content is an EmbedContentResponse object,
            # not a dictionary. Its embeddings are accessed via the .embeddings attribute.
            if not hasattr(response, 'embeddings') or not response.embeddings:
                logger.error(f"Gemini API response missing 'embeddings' attribute or it's empty. Full response: {response}")
                # Log any potential error messages if the response object has them
                if hasattr(response, 'error') and response.error: # Assuming 'error' could be an attribute
                     raise ValueError(f"Gemini API returned an error: {response.error.message if hasattr(response.error, 'message') else response.error}")
                else:
                    raise ValueError("Gemini API returned an unexpected response format (missing valid 'embeddings' attribute).")
            
            # Each item in response.embeddings is an Embedding object, which has a .value attribute
            embeddings.extend([item.value for item in response.embeddings])
            logger.info(f"Generated embeddings for batch {i//batch_size + 1}/{(len(texts) + batch_size - 1)//batch_size}")
        return embeddings
    except ValueError as ve: 
        logger.error(f"Error from Gemini API during embedding generation: {ve}")
        raise HTTPException(status_code=500, detail=f"Failed to generate embeddings: {ve}")
    except Exception as e:
        logger.error(f"An unexpected error occurred during Gemini embedding generation: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Failed to generate embeddings due to an unexpected error: {e}")

async def upsert_embeddings_to_pinecone(index, text_chunks: List[str], embeddings: List[List[float]], document_id: str):
    """Upserts embeddings and their corresponding text chunks (metadata) to Pinecone."""
    if not index:
        raise HTTPException(status_code=500, detail="Pinecone index is not initialized.")
    
    if len(text_chunks) != len(embeddings):
        logger.error(f"Mismatch in length: text_chunks={len(text_chunks)}, embeddings={len(embeddings)}")
        raise HTTPException(status_code=500, detail="Internal error: Mismatch between number of text chunks and generated embeddings during upsert preparation.")

    vectors_to_upsert = []
    for i, (chunk, embedding) in enumerate(zip(text_chunks, embeddings)):
        vector_id = f"{document_id}-chunk-{i}"
        vectors_to_upsert.append({
            "id": vector_id,
            "values": embedding,
            "metadata": {"text_chunk": chunk, "document_id": document_id, "chunk_index": i}
        })
    
    if not vectors_to_upsert:
        logger.warning("No vectors to upsert to Pinecone. This might mean no text was extracted or no embeddings were generated.")
        return 

    try:
        upsert_batch_size = 100 
        for i in range(0, len(vectors_to_upsert), upsert_batch_size):
            batch = vectors_to_upsert[i:i + upsert_batch_size]
            index.upsert(vectors=batch)
            logger.info(f"Upserted batch {i//upsert_batch_size + 1}/{(len(vectors_to_upsert) + upsert_batch_size - 1)//upsert_batch_size} to Pinecone.")
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
        raise HTTPException(status_code=500, detail="Pinecone index is not initialized.")

    if llm_client_type == "gemini" and not GEMINI_API_KEY:
        raise HTTPException(status_code=500, detail="GEMINI_KEY not set for Gemini LLM.")
    elif llm_client_type == "groq" and not groq_client:
        raise HTTPException(status_code=500, detail="Groq client not initialized for Groq LLM.")

    try:
        # 1. Generate embedding for the user's question
        query_embedding_response = genai.embed_content(
            model=GEMINI_EMBEDDING_MODEL,
            content=question,
            task_type="RETRIEVAL_QUERY"
        )
        
        # --- CRITICAL CHANGE HERE: Accessing embeddings attribute ---
        if not hasattr(query_embedding_response, 'embeddings') or not query_embedding_response.embeddings:
            logger.error(f"Gemini API response for query embedding missing 'embeddings' attribute or it's empty. Full response: {query_embedding_response}")
            if hasattr(query_embedding_response, 'error') and query_embedding_response.error:
                raise ValueError(f"Gemini API returned an error for query embedding: {query_embedding_response.error.message if hasattr(query_embedding_response.error, 'message') else query_embedding_response.error}")
            else:
                raise ValueError("Gemini API returned an unexpected response format for query embedding (missing valid 'embeddings' attribute).")

        query_embedding = query_embedding_response.embeddings[0].value # Access .value on the Embedding object

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
            model = genai.GenerativeModel(GEMINI_LLM_MODEL)
            response = await model.generate_content_async(prompt)
            answer = response.text
            logger.info(f"Gemini LLM answered question: '{question}'")
        elif llm_client_type == "groq":
            if not groq_client:
                raise HTTPException(status_code=500, detail="Groq client not initialized. Cannot use Groq LLM.")
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
    if not GEMINI_API_KEY:
        raise HTTPException(status_code=500, detail="GEMINI_KEY environment variable is not set. Cannot proceed.")
    if not pinecone_index:
        raise HTTPException(status_code=500, detail="Pinecone client failed to initialize. Check your Pinecone API key and environment.")

    logger.info(f"Received request to process PDF from: {request.documents}")
    logger.info(f"Questions received: {request.questions}")

    try:
        pdf_content = await download_pdf(request.documents)
        extracted_text = await extract_text_from_pdf(pdf_content)
        text_chunks = chunk_text(extracted_text, CHUNK_SIZE, CHUNK_OVERLAP)
        
        embeddings = await generate_embeddings(text_chunks)

        # The mismatch check is essential here. If generate_embeddings returns partial data
        # due to some internal Gemini issue or filtered content, this will catch it.
        if len(text_chunks) != len(embeddings):
            logger.error(f"Mismatch after embedding generation: text_chunks={len(text_chunks)}, embeddings={len(embeddings)}. Some chunks might not have received embeddings.")
            # Decide how to handle this:
            # 1. Raise HTTPException: Abort the process if embedding all chunks is critical.
            # 2. Filter text_chunks: Only proceed with chunks that have corresponding embeddings.
            #    For robust RAG, it's generally better to ensure all relevant chunks are embedded.
            #    Given the previous error, this might mean a problem with the API itself.
            raise HTTPException(status_code=500, detail="Internal error: Mismatch between number of text chunks and generated embeddings. Please check Gemini API status or input content.")
        
        document_id = str(uuid.uuid4()) 
        
        await upsert_embeddings_to_pinecone(pinecone_index, text_chunks, embeddings, document_id)
        
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