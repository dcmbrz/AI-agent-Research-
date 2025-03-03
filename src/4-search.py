'''
Searching:
– 4-search.py demonstrates querying the LanceDB table by searching for a keyword (like “pdf”)
 and returning relevant chunks based on their vector embeddings.
'''

import lancedb
import os
os.environ["OPENAI_API_KEY"] = ""

'''# connecting to the database
url= "data/lancedb"
db = lancedb.connect(url)


# loading the table
table = db.open_table("docling")


#searching the table
# query: the user question
# query_type: conducts a similarity search based on the query using the embeddings
result = table.search(query="pdf", query_type="vector").limit(5)
print(result.to_pandas())
'''
uri = "data/lancedb"
db = lancedb.connect(uri)


# --------------------------------------------------------------
# Load the table
# --------------------------------------------------------------

table = db.open_table("docling")


# --------------------------------------------------------------
# Search the table
# --------------------------------------------------------------

result = table.search(query="pdf").limit(5)
result.to_pandas()
