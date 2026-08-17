# Fine Print

RAG tool for asking questions about financial filings in plain English. LangChain + ChromaDB + OpenAI + Streamlit.

```
"Where did Capital One see the biggest gain in income and where did it invest heavily?"
→ splits into sub-questions, retrieves each separately, merges into one answer

"How do I build a Minecraft clone?"
→ rejected, out of scope
```

## Setup

```bash
python3 -m venv rag-env
source rag-env/bin/activate
pip install -r requirements.txt
```

`.env` file:
```
OPENAI_API_KEY=your-key-here
```

Drop PDFs into `data/`, then:

```bash
python3 ingest.py       # build the vectorstore
python3 query.py        # query from the CLI
streamlit run app.py    # web UI
```

## How it works

PDF → text extraction → boilerplate removal → chunking → OpenAI embeddings → ChromaDB → relevance-scored retrieval → GPT-3.5 for the final answer.

`ingest.py` handles cleaning/chunking/embedding. `query.py` handles retrieval and generation. `app.py` is the Streamlit front end.

## A few things worth explaining

I ran into a bug where a PDF's repeated footer text (its URL literally contained the words "net-income") was outscoring the real answer on a net-income question — the footer showed up on every page, so it dominated retrieval. Instead of just patching that one file, `ingest.py` now detects repeated header/footer lines automatically per document (lines that show up on most pages, near the top or bottom margins) and strips them before chunking, so any new PDF gets cleaned the same way without me writing new rules for it.

Compound questions were another issue — something like "where did income grow *and* where did they invest" embeds as one blended vector that doesn't match either topic well. `query.py` now asks the LLM to split a question like that into separate sub-questions first, retrieves for each one, and merges the results before answering.

Before any of that, retrieved chunks get a relevance score. If nothing scores high enough, the question gets rejected without ever calling the LLM — saves a call and keeps it from bluffing an answer out of irrelevant context.

## Known issues

- Re-ingesting requires manually clearing `vectorstore/` first (Chroma appends by default, doesn't overwrite)
- No source citations yet — can't tell you which doc/page an answer came from
- Rejection messages don't currently say what documents ARE available

## Stack

Python, LangChain, ChromaDB, OpenAI, Streamlit