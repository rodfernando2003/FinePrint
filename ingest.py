from dotenv import load_dotenv
import os
import re
from collections import defaultdict, Counter

from langchain_community.document_loaders import DirectoryLoader, PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_chroma import Chroma

load_dotenv()

# ---- Generic, format-agnostic boilerplate stripping ----
#
# Rather than hardcoding patterns for one specific PDF's header/footer
# style, we use two layers that generalize to ANY future document added
# to data/:
#
# 1. REGEX_PATTERNS: broad structural patterns common across most PDF
#    exports (timestamps, bare URLs, "page x/y" markers, lone footnote
#    numbers). These catch noise even when it varies slightly page to
#    page (e.g. a page number changes each page, so it can't be caught
#    by exact-duplicate detection alone).
#
# 2. Frequency-based duplicate detection: for each source PDF, count how
#    many pages each exact line appears on. A line repeated on most pages
#    of the SAME document (e.g. a running header/title, a footer URL, a
#    company name banner) is almost certainly boilerplate, regardless of
#    what it says. This adapts automatically to each new document's own
#    formatting without needing hand-written rules per file.
REGEX_PATTERNS = [
    r"^\d{1,2}/\d{1,2}/\d{2,4},?\s*\d{1,2}:\d{2}\s*[AP]M.*$",   # date/time header
    r"^https?://\S+$",                                          # bare URL lines
    r"^\d{1,3}\s*/\s*\d{1,3}$",                                 # bare page numbers e.g. "6/18"
    r"^\(\d+\)\s*\(\d+\)$",                                     # stray footnote markers e.g. "(8) (8)"
]
_compiled_regex = [re.compile(p) for p in REGEX_PATTERNS]

# A line must appear on at least this fraction of a document's pages to
# be treated as a repeated header/footer. Tune down if legitimate short
# repeated content is being missed, or up if real content is being stripped.
REPETITION_THRESHOLD = 0.5
MIN_PAGES_FOR_FREQUENCY_CHECK = 3  # skip frequency check on very short docs


def strip_regex_noise(text):
    lines = text.split("\n")
    kept = [
        line for line in lines
        if line.strip() and not any(p.match(line.strip()) for p in _compiled_regex)
    ]
    return "\n".join(kept)


def find_repeated_lines(pages_text):
    """Given a list of page texts (all from the same source document),
    return the set of exact lines that repeat across enough pages to be
    considered boilerplate."""
    if len(pages_text) < MIN_PAGES_FOR_FREQUENCY_CHECK:
        return set()

    line_page_counts = Counter()
    for text in pages_text:
        # Count each distinct line once per page (not per occurrence)
        unique_lines_this_page = {ln.strip() for ln in text.split("\n") if ln.strip()}
        for line in unique_lines_this_page:
            line_page_counts[line] += 1

    threshold_count = max(2, int(len(pages_text) * REPETITION_THRESHOLD))
    return {line for line, count in line_page_counts.items() if count >= threshold_count}


def clean_documents(documents):
    """Clean boilerplate from a list of langchain Document objects, grouped
    by source file so frequency detection is scoped per-document."""
    by_source = defaultdict(list)
    for doc in documents:
        source = doc.metadata.get("source", "unknown")
        by_source[source].append(doc)

    for source, docs in by_source.items():
        page_texts = [d.page_content for d in docs]
        repeated_lines = find_repeated_lines(page_texts)

        for doc in docs:
            lines = doc.page_content.split("\n")
            kept = [
                line for line in lines
                if line.strip() and line.strip() not in repeated_lines
            ]
            doc.page_content = strip_regex_noise("\n".join(kept))

    return documents


def ingest_documents():

    # Step 1 - Load documents from the data/ folder
    loader = DirectoryLoader("data/", glob="**/*.pdf", show_progress=True, loader_cls=PyPDFLoader)
    documents = loader.load()
    print(f"Loaded {len(documents)} documents.")

    # Step 1b - Strip repeated boilerplate before chunking. Works
    # automatically for any new PDF added to data/, since it detects
    # each document's own repeated headers/footers rather than relying
    # on rules written for one specific format.
    documents = clean_documents(documents)

    # Step 2 - Split documents into chunks
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    chunks = text_splitter.split_documents(documents)
    print('Split into', len(chunks), 'chunks.')

    # Step 3 - Create embeddings and store in ChromaDB
    print("Embedding and storing chunks...")
    embeddings = OpenAIEmbeddings()
    vectorstore = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory="vectorstore/"
    )

    print("Done. Vectorstore saved.")


if __name__ == "__main__":
    ingest_documents()