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

'ingest.py' handles cleaning/chunking/embedding. 'query.py' handles retrieval, citation building, and multi-turn conversation. 'app.py' is the Streamlit front end.

## A few things worth explaining

I ran into a bug early on where a PDF's repeated footer text (its URL literally contained the words "net-income") was outscoring the real answer on a net-income question — the footer showed up on every page, so it dominated retrieval. Instead of just patching that one file, 'ingest.py' now detects repeated header/footer lines automatically per document (lines that show up on most pages, near the top or bottom margins) and strips them before chunking, so any new PDF gets cleaned the same way without me writing new rules for it.

Compound questions were another issue — something like "where did income grow and where did they invest" embeds as one blended vector that doesn't match either topic well. 'query.py' now asks the LLM to split a question like that into separate sub-questions first, retrieves for each one, and merges the results before answering.

Every answer now cites which document and page it pulled from. That's built from the PDF's own metadata rather than trusted to the model's memory, so it can't just invent a page number that sounds right. I also added follow-up support — before retrieval runs, something like "how does that compare to last year" gets rewritten into a standalone question using the last few turns of conversation, since the retriever has no idea what "that" means on its own.

Before any of that happens, retrieved chunks get scored for relevance. If nothing scores high enough, the question gets rejected without ever calling the LLM — cheaper, faster, and it keeps the model from bluffing an answer out of context that isn't actually relevant.

## Known issues
- Dense financial tables are the biggest weak spot right now. They get flattened into plain text on ingestion, and I've caught the model misreading which number belongs to which quarter or business segment in a wide table — once it cited a Consumer Banking figure as if it were the company-wide total. Citations make this catchable (always check the cited page if a number matters), but they don't prevent it. If I had more time I'd extract tables as structured data at ingestion instead of flattened text, which should fix it properly.
- Re-ingesting means manually clearing 'vectorstore/' first, since Chroma appends instead of overwriting by default.
- Rejection messages tell you which documents are loaded but not what's actually in them yet.

## Stack
Python, LangChain, ChromaDB, OpenAI, Streamlit