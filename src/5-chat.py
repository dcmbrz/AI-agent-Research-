'''
Interactive Chat Interface:
– 5-chat.py provides a web-based (Streamlit) interface for document Q&A.
  It retrieves context by querying the "docling" table, then feeds that context along
  with the user’s question to the OpenAI chat model to generate a response.
--------------------------------------------------------------------------------------------
How to run:
streamlit run src/5-chat.py
'''

import streamlit as st
import lancedb
import openai
from dotenv import load_dotenv
import os
import tempfile  # Added import for temporary file handling

# Patch openai to provide an OpenAI client for lancedb
if not hasattr(openai, "OpenAI"):
    class DummyOpenAIClient:
        def __init__(self, **kwargs):
            self.api_key = kwargs.get("api_key", None)
        @property
        def embeddings(self):
            class EmbeddingsWrapper:
                def create(inner_self, **kwargs):
                    # Call the new API for embeddings.
                    # Note: lancedb might pass parameters expected by the old API.
                    # Adjust as necessary if there are model or parameter differences.
                    return openai.Embedding.create(**kwargs)
            return EmbeddingsWrapper()
    openai.OpenAI = DummyOpenAIClient


# Document Ingestion-related imports
from docling.document_converter import DocumentConverter
from docling.chunking import HybridChunker
from utils.tokenizer import OpenAITokenizerWrapper
from lancedb.embeddings import get_registry
from lancedb.pydantic import LanceModel, Vector


# Load environment variables
load_dotenv()
API_KEY = os.getenv("OPENAI_API_KEY")
if not API_KEY:
    raise ValueError("Missing OPENAI_API_KEY environment variable")
openai.api_key = API_KEY

# SideBar for Mode Selection (Chat or Document Ingestion)
st.sidebar.title("Options")
mode = st.sidebar.radio("Select Mode", ["Upload Document", "Chat"])

# ----------------------------
# Document Ingestion Section
# ----------------------------
if mode == "Upload Document":
    st.header("Upload a New Document")
    doc_source = st.radio("Select the Document's Source:", ["URL", "PDF Upload"])
    converter = DocumentConverter()
    result = None

    if doc_source == "URL":
        url = st.text_input("Enter the document URL:")
        if st.button("Process URL") and url:
            with st.spinner("Processing document from URL..."):
                result = converter.convert(url)
    else:
        uploaded_file = st.file_uploader("Upload a PDF file", type=["pdf"])
        if st.button("Process Upload") and uploaded_file is not None:
            with st.spinner("Processing uploaded PDF..."):
                # Write the uploaded file to a temporary file
                with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
                    tmp_file.write(uploaded_file.getvalue())
                    tmp_path = tmp_file.name  # Get the file path as a string
                result = converter.convert(tmp_path)

    if result is not None:
        st.success("Document processed successfully!")

        # Proceed to chunk and ingest the documents:
        tokenizer = OpenAITokenizerWrapper()
        _MAX_TOKENS = 8191
        chunker = HybridChunker(tokenizer=tokenizer, max_tokens=_MAX_TOKENS, merge_peers=True)
        chunk_iter = chunker.chunk(dl_doc=result.document)
        chunks = list(chunk_iter)
        st.write(f"Extracted {len(chunks)} chunks.")

        # Connect to the lancedb using the same absolute path logic
        BASE_DIR = os.path.dirname(os.path.abspath(__file__))
        db_path = os.path.join(BASE_DIR, "..", "data", "lancedb")
        db = lancedb.connect(db_path)

        # same embedding function
        func = get_registry().get("openai").create(name="text-embedding-3-large")

        # Defining the schema classes if they aren't already defined
        class ChunkMetadata(LanceModel):
            filename: str | None
            page_numbers: list[int] | None
            title: str | None

        class Chunks(LanceModel):
            text: str = func.SourceField()
            vector: Vector(func.ndims()) = func.VectorField()
            metadata: ChunkMetadata

        # Opening the existing table
        try:
            table = db.open_table("docling")
        except Exception as e:
            st.error(f"Error opening table: {e}")
            st.stop()

        # preparing processed chunks with metadata
        new_processed_chunks = []
        for chunk in chunks:
            metadata = {
                "filename": chunk.meta.origin.filename if hasattr(chunk.meta, "origin") else None,
                "page_numbers": (
                    sorted(set(prov.page_no for item in chunk.meta.doc_items for prov in item.prov))
                    if hasattr(chunk.meta, "doc_items") else None
                ),
                "title": chunk.meta.headings[0] if chunk.meta.headings else None
            }
            new_processed_chunks.append({"text": chunk.text, "metadata": metadata})

        # Append the new document's chunks to the table
        table.add(new_processed_chunks)
        st.success("New document ingested into the database!")
        st.stop()  # Stop further processing, so user can switch back to Chat mode.

# ----------------------------
# Chat Interface Section
# ----------------------------
else:
    @st.cache_resource
    def init_db():
        BASE_DIR = os.path.dirname(os.path.abspath(__file__))
        db_path = os.path.join(BASE_DIR, "..", "data", "lancedb")
        st.write("Connecting to LanceDB at:", db_path)  # Debug print
        db = lancedb.connect(db_path)
        return db.open_table("docling")

    def get_context(query: str, table, num_results: int = 3) -> str:
        results = table.search(query).limit(num_results).to_pandas()
        contexts = []
        for _, row in results.iterrows():
            filename = row["metadata"]["filename"]
            page_numbers = row["metadata"]["page_numbers"]
            title = row["metadata"]["title"]
            source_parts = []
            if filename:
                source_parts.append(filename)
            if page_numbers:
                source_parts.append(f"p. {', '.join(str(p) for p in page_numbers)}")
            source = f"\nSource: {' - '.join(source_parts)}"
            if title:
                source += f"\nTitle: {title}"
            contexts.append(f"{row['text']}{source}")
        return "\n\n".join(contexts)

    def get_chat_response(messages, context: str) -> str:
        system_prompt = f"""You are a helpful assistant that answers questions based on the provided context.
Use only the information from the context to answer questions. If you're unsure or the context
doesn't contain the relevant information, say so.

Context:
{context}
"""
        messages_with_context = [{"role": "system", "content": system_prompt}, *messages]
        # Create the response (for debugging, we won't stream)
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=messages_with_context,
            temperature=0.7,
        )
        st.set_page_config(page_title="Article Analysis Agent", page_icon=":mortar_board")
        # Extract full response text from choices (assuming one choice)
        full_response = response["choices"][0]["message"]["content"]
        st.write("DEBUG: Chat response:", full_response)  # Debug print
        return full_response

    # Debug: Announce that Chat mode is active
    st.write("Chat mode active.")

    # For debugging, use st.text_input
    prompt = st.text_input("Ask a question about the document:")

    # Initialize session state for chat history if not already set
    if "messages" not in st.session_state:
        st.session_state.messages = []

    # Initialize the database (table)
    table = init_db()

    # Display previous chat messages (using st.write for debugging)
    st.write("Chat History:")
    for message in st.session_state.messages:
        st.write(f"{message['role']}: {message['content']}")

    if prompt:
        # Append user message
        st.session_state.messages.append({"role": "user", "content": prompt})
        st.write("User:", prompt)

        # Retrieve context from the database
        with st.spinner("Searching document..."):
            context = get_context(prompt, table)
            st.write("Relevant Document Sections:")
            st.write(context)

        # Get chat response
        with st.spinner("Getting response..."):
            response = get_chat_response(st.session_state.messages, context)

        st.write("Assistant:", response)
        st.session_state.messages.append({"role": "assistant", "content": response})
