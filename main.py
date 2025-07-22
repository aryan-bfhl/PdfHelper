import os
import io
import requests
import PyPDF2
import google.generativeai as genai
from pinecone import Pinecone, Index
from groq import Groq
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Dict, Tuple
import logging
import uuid

# Configure logging for better visibility
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# --- Configuration ---
# IMPORTANT: Set these environment variables before running the application.
# Example:
# export GEMINI_API_KEY="YOUR_GEMINI_API_KEY"
# export PINECONE_API_KEY="YOUR_PINECONE_API_KEY"
# export PINECONE_ENVIRONMENT="YOUR_PINECONE_ENVIRONMENT" # e.g., 'us-west-2' or 'gcp-starter'
# export GROQ_API_KEY="YOUR_GROQ_API_KEY" # Optional: Only if you want to use Groq LLM

GEMINI_API_KEY = os.getenv("GEMINI_KEY")
PINECONE_API_KEY = os.getenv("PINECONE_KEY")
PINECONE_ENVIRONMENT = os.getenv("PINECONE_ENV")
GROQ_API_KEY = os.getenv("GROQ_KEY")

# Pinecone index configuration
PINECONE_INDEX_NAME = "new"
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
    logger.error("GEMINI_API_KEY environment variable not set. Gemini services will not be available.")

# Initialize Pinecone client and index
pinecone_client = None
pinecone_index = None
if PINECONE_API_KEY and PINECONE_ENVIRONMENT:
    try:
        pinecone_client = Pinecone(api_key=PINECONE_API_KEY, environment=PINECONE_ENVIRONMENT)
        
        # Check if the Pinecone index exists, create it if not
        if PINECONE_INDEX_NAME not in pinecone_client.list_indexes():
            logger.info(f"Creating Pinecone index: {PINECONE_INDEX_NAME} with dimension {EMBEDDING_DIMENSION}")
            # For serverless, specify cloud and region. Adjust 'spec' according to your Pinecone account type.
            pinecone_client.create_index(
                name=PINECONE_INDEX_NAME,
                dimension=EMBEDDING_DIMENSION,
                metric='cosine', # Cosine similarity is common for embeddings
                spec={"serverless": {"cloud": "aws", "region": "us-west-2"}} # Example for AWS serverless
                # For pod-based, use: spec={"pod": {"environment": PINECONE_ENVIRONMENT, "pod_type": "p1.x1", "pods": 1}}
            )
        pinecone_index = pinecone_client.Index(PINECONE_INDEX_NAME)
        logger.info(f"Connected to Pinecone index: {PINECONE_INDEX_NAME}")
    except Exception as e:
        logger.error(f"Failed to initialize Pinecone. Please check PINECONE_API_KEY and PINECONE_ENVIRONMENT: {e}")
else:
    logger.error("PINECONE_API_KEY or PINECONE_ENVIRONMENT not set. Pinecone services will not be available.")

# Initialize Groq client (optional)
groq_client = None
if GROQ_API_KEY:
    try:
        groq_client = Groq(api_key=GROQ_API_KEY)
        logger.info("Groq client initialized.")
    except Exception as e:
        logger.warning(f"Failed to initialize Groq client: {e}. Groq LLM will not be available.")

# --- Pydantic Models for API Request/Response ---
class ProcessRequest(BaseModel):
    blob_link: str # URL to the PDF document
    questions: List[str] # List of questions to ask about the PDF

class ProcessResponse(BaseModel):
    answers: List[str] # List of answers corresponding to the questions

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
    This function is designed for text-based PDFs. Since the requirement specifies
    "there won't be image-based PDFs", direct OCR via `pytesseract` and `pdf2image`
    is explicitly excluded.
    """
    text = ""
    try:
        pdf_file = io.BytesIO(pdf_content)
        reader = PyPDF2.PdfReader(pdf_file)
        
        # Check if the PDF has any pages
        if not reader.pages:
            raise HTTPException(status_code=400, detail="The PDF document appears to be empty or corrupted.")

        for page_num in range(len(reader.pages)):
            page = reader.pages[page_num]
            extracted_page_text = page.extract_text()
            if extracted_page_text:
                text += extracted_page_text + "\n" # Add newline for better separation
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
        chunk = text[start:min(end, len(text))] # Ensure not to go past text length
        chunks.append(chunk)
        if end >= len(text): # Stop if we've reached the end
            break
        start += chunk_size - chunk_overlap # Move start for next chunk with overlap
    logger.info(f"Text chunked into {len(chunks)} pieces.")
    return chunks

async def generate_embeddings(texts: List[str]) -> List[List[float]]:
    """Generates embeddings for a list of text chunks using Gemini's embedding model."""
    if not GEMINI_API_KEY:
        raise HTTPException(status_code=500, detail="Gemini API key not configured.")
    
    embeddings = []
    try:
        # Gemini's embed_content can handle multiple texts, but it's good practice
        # to batch requests for very large lists to avoid potential rate limits or timeouts.
        # The actual batch size limit can vary, 100 is a safe starting point.
        batch_size = 100 
        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i:i + batch_size]
            response = genai.embed_content(
                model=GEMINI_EMBEDDING_MODEL,
                content=batch_texts,
                task_type="RETRIEVAL_DOCUMENT" # Specify task type for better embedding quality
            )
            # The response contains 'embeddings' where each item has a 'value' key
            embeddings.extend([item['value'] for item in response['embeddings']])
            logger.info(f"Generated embeddings for batch {i//batch_size + 1}/{(len(texts) + batch_size - 1)//batch_size}")
        return embeddings
    except Exception as e:
        logger.error(f"Error generating embeddings with Gemini: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to generate embeddings: {e}")

async def upsert_embeddings_to_pinecone(index: Index, text_chunks: List[str], embeddings: List[List[float]], document_id: str):
    """Upserts embeddings and their corresponding text chunks (metadata) to Pinecone."""
    if not index:
        raise HTTPException(status_code=500, detail="Pinecone index is not initialized.")
    
    vectors_to_upsert = []
    for i, (chunk, embedding) in enumerate(zip(text_chunks, embeddings)):
        # Create a unique ID for each vector, combining document ID and chunk index
        vector_id = f"{document_id}-chunk-{i}"
        vectors_to_upsert.append({
            "id": vector_id,
            "values": embedding,
            "metadata": {"text_chunk": chunk, "document_id": document_id, "chunk_index": i}
        })
    
    try:
        # Pinecone upsert operations are also batched for efficiency and to respect limits
        upsert_batch_size = 100 # Adjust based on Pinecone limits and network conditions
        for i in range(0, len(vectors_to_upsert), upsert_batch_size):
            batch = vectors_to_upsert[i:i + upsert_batch_size]
            index.upsert(vectors=batch)
            logger.info(f"Upserted batch {i//upsert_batch_size + 1}/{(len(vectors_to_upsert) + upsert_batch_size - 1)//upsert_batch_size} to Pinecone.")
        logger.info(f"Successfully upserted {len(vectors_to_upsert)} embeddings to Pinecone for document {document_id}.")
    except Exception as e:
        logger.error(f"Error upserting embeddings to Pinecone: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to upsert embeddings to Pinecone: {e}")

async def query_pinecone_and_rag(index: Index, question: str, llm_client_type: str = "gemini", top_k: int = 3) -> str:
    """
    Queries Pinecone for relevant context based on the question,
    then uses an LLM (Gemini or Groq) to answer the question with RAG.
    `llm_client_type` can be "gemini" or "groq".
    """
    if not index:
        raise HTTPException(status_code=500, detail="Pinecone index is not initialized.")

    if llm_client_type == "gemini" and not GEMINI_API_KEY:
        raise HTTPException(status_code=500, detail="GEMINI_API_KEY not set for Gemini LLM.")
    elif llm_client_type == "groq" and not groq_client:
        raise HTTPException(status_code=500, detail="Groq client not initialized for Groq LLM.")

    try:
        # 1. Generate embedding for the user's question
        query_embedding_response = genai.embed_content(
            model=GEMINI_EMBEDDING_MODEL,
            content=question,
            task_type="RETRIEVAL_QUERY" # Specify task type for query embeddings
        )
        query_embedding = query_embedding_response['embeddings'][0]['value']

        # 2. Query Pinecone to find top_k most relevant text chunks
        query_results = index.query(
            vector=query_embedding,
            top_k=top_k,
            include_metadata=True # Important to retrieve the original text chunks
        )
        
        context_chunks = [match.metadata['text_chunk'] for match in query_results.matches if 'text_chunk' in match.metadata]
        
        if not context_chunks:
            logger.warning(f"No relevant context found in Pinecone for question: '{question}'. Answering without specific context.")
            context_string = "No specific context found in the document."
        else:
            # Combine retrieved chunks into a single context string
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
            # Use generate_content_async for async FastAPI endpoint
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
                temperature=0.7, # Adjust temperature for creativity vs. factualness
                max_tokens=500,  # Max tokens for the LLM's response
            )
            answer = chat_completion.choices[0].message.content
            logger.info(f"Groq LLM answered question: '{question}'")
        else:
            raise ValueError("Invalid LLM client type specified. Must be 'gemini' or 'groq'.")

        return answer

    except Exception as e:
        logger.error(f"Error during RAG process for question '{question}': {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Failed to answer question using RAG: {e}")

# --- FastAPI Endpoint ---
@app.post("/process_pdf_questions", response_model=ProcessResponse)
async def process_pdf_questions_endpoint(request: ProcessRequest):
    """
    API endpoint to process a PDF from a blob link, generate embeddings,
    and answer a list of questions using RAG.
    """
    # Pre-check for essential service availability
    if not GEMINI_API_KEY:
        raise HTTPException(status_code=500, detail="GEMINI_API_KEY environment variable is not set. Cannot proceed.")
    if not pinecone_index:
        raise HTTPException(status_code=500, detail="Pinecone client failed to initialize. Check your Pinecone API key and environment.")

    logger.info(f"Received request to process PDF from: {request.blob_link}")
    logger.info(f"Questions received: {request.questions}")

    try:
        # 1. Download the PDF content
        pdf_content = await download_pdf(request.blob_link)
        
        # 2. Extract text from the PDF (assuming it's text-based)
        extracted_text = await extract_text_from_pdf(pdf_content)
        
        # 3. Chunk the extracted text into smaller, manageable pieces
        text_chunks = chunk_text(extracted_text, CHUNK_SIZE, CHUNK_OVERLAP)
        
        # 4. Generate embeddings for each text chunk using Gemini
        embeddings = await generate_embeddings(text_chunks)

        if len(text_chunks) != len(embeddings):
            raise HTTPException(status_code=500, detail="Internal error: Mismatch between number of text chunks and generated embeddings.")
        
        # Generate a unique document ID for this PDF processing session
        document_id = str(uuid.uuid4())
        
        # 5. Upsert the generated embeddings and their metadata to Pinecone
        await upsert_embeddings_to_pinecone(pinecone_index, text_chunks, embeddings, document_id)
        
        # 6. Process each question using RAG and collect answers
        answers = []
        for question in request.questions:
            # You can choose "gemini" or "groq" for the LLM here.
            # Defaulting to "gemini" as it's part of the Gemini SDK request.
            # If you want to use Groq, ensure GROQ_API_KEY is set and change to "groq".
            answer = await query_pinecone_and_rag(pinecone_index, question, llm_client_type="gemini") 
            answers.append(answer)
        
        logger.info("Successfully processed PDF and answered all questions.")
        return ProcessResponse(answers=answers)

    except HTTPException as e:
        # Re-raise HTTPExceptions as they contain specific status codes and details
        logger.error(f"API Error: {e.detail}", exc_info=True)
        raise e
    except Exception as e:
        # Catch any other unexpected errors and return a generic 500 error
        logger.error(f"An unhandled error occurred during PDF processing: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"An internal server error occurred: {e}")

# --- How to Run This API ---
# 1. Save the code: Save this content as `main.py` (or any other .py file).
#
# 2. Install dependencies:
#    pip install fastapi uvicorn requests PyPDF2 google-generativeai pinecone-client groq pydantic
#
# 3. Set Environment Variables:
#    Before running, set your API keys as environment variables in your terminal:
#    export GEMINI_API_KEY="YOUR_GEMINI_API_KEY"
#    export PINECONE_API_KEY="YOUR_PINECONE_API_KEY"
#    export PINECONE_ENVIRONMENT="YOUR_PINECONE_ENVIRONMENT" # e.g., 'us-west-2' or 'gcp-starter'
#    export GROQ_API_KEY="YOUR_GROQ_API_KEY" # Optional, if you want to use Groq
#
# 4. Run the API:
#    uvicorn main:app --host 0.0.0.0 --port 8000
#
# 5. Test the API (e.g., using curl or Postman):
#    curl -X POST "http://localhost:8000/process_pdf_questions" \
#    -H "Content-Type: application/json" \
#    -d '{
#      "blob_link": "https://www.africau.edu/images/default/sample.pdf",
#      "questions": ["What is the main topic of the document?", "Who is the author?"]
#    }'
