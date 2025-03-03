'''
Chunking:
–  2-chunking.py breaks a document into smaller pieces (chunks)
 using a hybrid chunker, preparing the text for embedding.
'''
import os
from docling.chunking import HybridChunker
from docling.document_converter import DocumentConverter
from dotenv import load_dotenv
import openai  # Import the module, not just a Client class
from utils.tokenizer import OpenAITokenizerWrapper

# Load environment variables from .env
load_dotenv()
API_KEY = os.getenv("OPENAI_API_KEY")
if not API_KEY:
    raise ValueError("No API key found in environment variable OPENAI_API_KEY.")

# Set the API key globally for the openai module
openai.api_key = API_KEY

# Initialize our custom tokenizer for OpenAI
tokenizer = OpenAITokenizerWrapper()
MAX_TOKENS = 8191  # text-embedding-3-large's maximum context length

# --------------------------------------------------------------
# Extract the data
# --------------------------------------------------------------
converter = DocumentConverter()
result = converter.convert("https://arxiv.org/pdf/2408.09869")

# --------------------------------------------------------------
# Apply hybrid chunking
# --------------------------------------------------------------
chunker = HybridChunker(
    tokenizer=tokenizer,
    max_tokens=MAX_TOKENS,
    merge_peers=True,
)

chunk_iter = chunker.chunk(dl_doc=result.document)
chunks = list(chunk_iter)

print("chunks: ", chunks)
print("number of chunks:", len(chunks))
