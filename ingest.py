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
# Three layers of protection, each guarding against a different edge case:
#
# 1. REGEX_PATTERNS - broad structural noise common across most PDF
#    exports (timestamps, bare URLs, page-number markers). Catches noise
#    that varies slightly page to page (a page number changes each page),
#    so exact-duplicate detection alone wouldn't catch it.
#
# 2. Position-aware frequency detection - a line repeated across most
#    pages of the SAME document is treated as boilerplate, but ONLY if it
#    appears near the top or bottom of the page (headers/footers live at
#    page edges). This protects repeated TABLE content in the middle of a
#    page - e.g. a column header row like "Q1 Q4 Q3 Q2 Q1" that
#    legitimately recurs across many pages of a financial table - since
#    that kind of repetition happens in the page body, not the margins.
#
# 3. Per-page safety cap - even if many lines on a page match, never
#    strip more than a set fraction of that page's lines. This protects
#    SHORT documents or table-heavy pages from being gutted if the
#    heuristics misfire, since a single bad match matters more when
#    there's little content to begin with.
REGEX_PATTERNS = [
    r"^\d{1,2}/\d{1,2}/\d{2,4},?\s*\d{1,2}:\d{2}\s*[AP]M.*$",   # date/time header
    r"^https?://\S+$",                                          # bare URL lines
    r"^\d{1,3}\s*/\s*\d{1,3}$",                                 # bare page numbers e.g. "6/18"
    r"^\(\d+\)\s*\(\d+\)$",                                     # stray footnote markers e.g. "(8) (8)"
]
_compiled_regex = [re.compile(p) for p in REGEX_PATTERNS]

# A line must appear on at least this fraction of a document's pages to
# be treated as a repeated header/footer.
REPETITION_THRESHOLD = 0.5

# Very short documents get a stricter (higher) threshold instead of being
# skipped outright, since a coincidental repeat across 2-3 pages is more
# likely with less data to average over.
MIN_PAGES_FOR_FREQUENCY_CHECK = 2
SHORT_DOC_PAGE_COUNT = 3
SHORT_DOC_THRESHOLD = 0.9  # require near-total repetition on short docs

# Only lines within this many positions of the page's top or bottom are
# eligible for frequency-based removal - protects repeated table rows
# that live in the middle of a page.
MARGIN_LINE_COUNT = 3

# Never remove more than this fraction of a single page's lines, even if
# more lines technically match - a safety net against over-stripping
# short or table-heavy pages.
MAX_REMOVAL_FRACTION_PER_PAGE = 0.4

# If True, print a summary of what was removed per source file so you can
# spot-check a new document type the first time you ingest it.
AUDIT_LOG = True


def strip_regex_noise(lines):
    return [
        line for line in lines
        if not any(p.match(line.strip()) for p in _compiled_regex)
    ]


def find_repeated_lines(pages_lines):
    """Given a list of per-page line lists (all from the same source
    document), return the set of exact lines - restricted to each page's
    margins - that repeat across enough pages to be considered
    boilerplate."""
    if len(pages_lines) < MIN_PAGES_FOR_FREQUENCY_CHECK:
        return set()

    threshold = SHORT_DOC_THRESHOLD if len(pages_lines) <= SHORT_DOC_PAGE_COUNT else REPETITION_THRESHOLD

    line_page_counts = Counter()
    for lines in pages_lines:
        margin_lines = set(lines[:MARGIN_LINE_COUNT]) | set(lines[-MARGIN_LINE_COUNT:])
        for line in margin_lines:
            if line.strip():
                line_page_counts[line.strip()] += 1

    threshold_count = max(2, int(len(pages_lines) * threshold))
    return {line for line, count in line_page_counts.items() if count >= threshold_count}


def clean_documents(documents):
    """Clean boilerplate from a list of langchain Document objects, grouped
    by source file so frequency detection is scoped per-document."""
    by_source = defaultdict(list)
    for doc in documents:
        source = doc.metadata.get("source", "unknown")
        by_source[source].append(doc)

    for source, docs in by_source.items():
        pages_lines = [
            [ln for ln in d.page_content.split("\n") if ln.strip()]
            for d in docs
        ]
        repeated_lines = find_repeated_lines(pages_lines)

        total_removed = 0
        total_lines = 0

        for doc, lines in zip(docs, pages_lines):
            total_lines += len(lines)
            candidate_removed = [ln for ln in lines if ln.strip() in repeated_lines]

            # Safety cap: if removing all matched lines would strip more
            # than the allowed fraction of this page, keep only enough
            # matches (favoring margin positions first) to stay under cap.
            max_removable = int(len(lines) * MAX_REMOVAL_FRACTION_PER_PAGE)
            if len(candidate_removed) > max_removable:
                # Keep the page mostly intact rather than over-stripping;
                # skip frequency removal for this page and rely on regex only.
                kept = strip_regex_noise(lines)
            else:
                kept = strip_regex_noise([ln for ln in lines if ln.strip() not in repeated_lines])

            total_removed += len(lines) - len(kept)
            doc.page_content = "\n".join(kept)

        if AUDIT_LOG and total_lines > 0:
            pct = (total_removed / total_lines) * 100
            print(f"  [cleaned] {os.path.basename(source)}: removed {total_removed}/{total_lines} lines ({pct:.1f}%)")
            if repeated_lines:
                sample = list(repeated_lines)[:3]
                print(f"    sample boilerplate detected: {sample}")

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