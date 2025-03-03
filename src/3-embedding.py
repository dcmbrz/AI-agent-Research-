'''
Embedding & Storage:
– 3-embedding.py takes these chunks, computes embeddings via OpenAI’s model (through LanceDB’s integration),
and stores the results along with metadata in a LanceDB table named "docling". This table becomes your searchable index.
'''

import os
from typing import List
import lancedb
from docling.chunking import HybridChunker
from docling.document_converter import DocumentConverter
from dotenv import load_dotenv
from lancedb.embeddings import get_registry
from lancedb.pydantic import LanceModel, Vector
import openai  # Import the openai module
from utils.tokenizer import OpenAITokenizerWrapper

# Load environment variables from .env file
load_dotenv()

# Retrieve the API key from environment variable
API_KEY = os.getenv("OPENAI_API_KEY")
if not API_KEY:
    raise ValueError("Missing OPENAI_API_KEY environment variable.")

# Set the OpenAI API key globally
openai.api_key = API_KEY

# (Optional) Remove explicit Client instantiation unless needed elsewhere
# from openai import Client
# client = Client(api_key=API_KEY)

tokenizer = OpenAITokenizerWrapper()  # Load our custom tokenizer for OpenAI
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

# --------------------------------------------------------------
# Create a LanceDB database and table
# --------------------------------------------------------------
db = lancedb.connect("data/lancedb")

# Get the OpenAI embedding function
func = get_registry().get("openai").create(name="text-embedding-3-large")

# Define a simplified metadata schema
class ChunkMetadata(LanceModel):
    """
    You must order the fields in alphabetical order.
    This is a requirement of the Pydantic implementation.
    """
    filename: str | None
    page_numbers: List[int] | None
    title: str | None

# Define the main schema for the chunks
class Chunks(LanceModel):
    text: str = func.SourceField()  # Source field sent to the embedding model
    vector: Vector(func.ndims()) = func.VectorField()  # Vector column for embeddings
    metadata: ChunkMetadata

# Create (or overwrite) the table named "docling"
table = db.create_table("docling", schema=Chunks, mode="overwrite")

# --------------------------------------------------------------
# Prepare the chunks for the table
# --------------------------------------------------------------
processed_chunks = [
    {
        "text": chunk.text,
        "metadata": {
            "filename": chunk.meta.origin.filename,
            "page_numbers": [
                page_no
                for page_no in sorted(
                    set(
                        prov.page_no
                        for item in chunk.meta.doc_items
                        for prov in item.prov
                    )
                )
            ] or None,
            "title": chunk.meta.headings[0] if chunk.meta.headings else None,
        },
    }
    for chunk in chunks
]

# --------------------------------------------------------------
# Add the chunks to the table (automatically embeds the text)
# --------------------------------------------------------------
table.add(processed_chunks)

# --------------------------------------------------------------
# Load and inspect the table
# --------------------------------------------------------------
print("Number of rows in table:", table.count_rows())
print(table.to_pandas())
#print(db.list_tables())
