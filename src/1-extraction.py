'''
Extraction & Conversion:
– 1-extraction.py converts PDFs and HTML pages into document objects,
 and optionally scrapes multiple pages using a sitemap.
'''



from docling.document_converter import DocumentConverter
from utils.sitemap import get_sitemap_urls


converter = DocumentConverter()

# --------------------------------------------------------------
# Basic PDF extraction
# --------------------------------------------------------------


result = converter.convert("https://arxiv.org/pdf/2408.09869")

document = result.document
markdown_output = document.export_to_markdown()
json_output = document.export_to_dict()
print("|-----------------------PFD MARKDOWN-----------------------------|", "\n")
print(markdown_output, "\n")

# --------------------------------------------------------------
# Basic HTML extraction
# --------------------------------------------------------------

result = converter.convert("https://ds4sd.github.io/docling/")

document = result.document
markdown_output = document.export_to_markdown()
print("|-----------------------WEBSTIE MARKDOWN-----------------------------|", "\n")
print(markdown_output, "\n")

# --------------------------------------------------------------
# Scrape multiple pages using the sitemap
# --------------------------------------------------------------

sitemap_urls = get_sitemap_urls("https://ds4sd.github.io/docling/")
conv_results_iter = converter.convert_all(sitemap_urls)

docs = []
for result in conv_results_iter:
    if result.document:
        document = result.document
        docs.append(document)


