from dotenv import load_dotenv
import os

from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_chroma import Chroma
from langchain_core.prompts import ChatPromptTemplate

load_dotenv()

# Below this relevance score, we treat the question as out-of-scope
# and skip the LLM call entirely rather than let it guess from
# irrelevant chunks. Tune this based on false positives/negatives
# you observe (higher = stricter).
RELEVANCE_THRESHOLD = 0.3

# How many prior (question, answer) turns to include when rewriting a
# follow-up question and when giving the final answer conversational
# context. Keeping this small controls token usage/cost as a conversation
# grows - older turns matter less for resolving "that" or "it" than the
# most recent one or two exchanges.
MAX_HISTORY_TURNS = 4

GENERIC_FALLBACK = "I don't have enough information in the provided documents to answer that."


def get_available_sources(vectorstore):
    """Pull distinct source filenames from the vectorstore so refusal
    messages can tell the user what's actually in scope, instead of a
    generic 'I don't know.'"""
    try:
        all_docs = vectorstore.get(include=["metadatas"])
        sources = {
            os.path.basename(m.get("source", "unknown"))
            for m in all_docs.get("metadatas", []) if m
        }
        return sorted(sources)
    except Exception:
        return []


def build_fallback_message(vectorstore):
    """Dynamic fallback for the hard relevance guard (question is so
    off-topic the LLM is never even called). Tells the user what
    documents are actually available rather than a bare refusal."""
    sources = get_available_sources(vectorstore)
    if not sources:
        return GENERIC_FALLBACK
    doc_list = ", ".join(sources)
    return (
        f"{GENERIC_FALLBACK} I currently have access to: {doc_list}."
    )


def format_context_with_citations(docs):
    """Number each retrieved chunk and label it with its source file and
    page number (pulled from PyPDFLoader metadata). The LLM is asked to
    cite these numbers inline; the actual file/page mapping is built here
    programmatically rather than trusted to the LLM, since it could
    otherwise invent a page number that sounds plausible but is wrong."""
    blocks = []
    citation_map = {}
    for i, doc in enumerate(docs, start=1):
        source = os.path.basename(doc.metadata.get("source", "unknown"))
        page = doc.metadata.get("page")
        page_display = f"page {page + 1}" if isinstance(page, int) else "page unknown"
        citation_map[i] = f"{source}, {page_display}"
        blocks.append(f"[{i}] (Source: {source}, {page_display})\n{doc.page_content}")
    return "\n\n".join(blocks), citation_map


def format_sources_footer(citation_map, answer_text):
    """Only list sources whose citation number actually appears in the
    answer text, so the footer reflects what was actually used rather
    than everything that was retrieved."""
    used = sorted(
        {num for num in citation_map if f"[{num}]" in answer_text}
    )
    if not used:
        return ""
    lines = ["\nSources:"]
    for num in used:
        lines.append(f"  [{num}] {citation_map[num]}")
    return "\n".join(lines)


def format_history_for_prompt(chat_history):
    """Render recent (question, answer) turns as plain text for prompts.
    Strips any citation/source footers from prior answers so they don't
    confuse the rewriting or generation steps with old citation numbers."""
    trimmed = chat_history[-MAX_HISTORY_TURNS:]
    lines = []
    for q, a in trimmed:
        clean_answer = a.split("\nSources:")[0].strip()
        lines.append(f"User: {q}\nAssistant: {clean_answer}")
    return "\n\n".join(lines)


def rewrite_with_history(question, chat_history, llm):
    """Follow-up questions like 'how does that compare to last year' can't
    be meaningfully embedded on their own - the retriever has no idea what
    'that' refers to. This rewrites the question into a standalone version
    using recent conversation history, so retrieval has something concrete
    to search for. If there's no history yet, or the question is already
    standalone, this is a no-op (the LLM is instructed to return it as-is).
    Not streamed - this is a short, internal step the user never sees."""
    if not chat_history:
        return question

    history_text = format_history_for_prompt(chat_history)
    rewrite_prompt = (
        "Given the conversation history and a follow-up question, rewrite the "
        "follow-up question as a standalone question that includes all context "
        "needed to understand it without the history. If it's already standalone, "
        "return it unchanged. Return ONLY the rewritten question, nothing else.\n\n"
        f"Conversation history:\n{history_text}\n\n"
        f"Follow-up question: {question}\n\n"
        "Standalone question:"
    )
    rewritten = llm.invoke(rewrite_prompt).content.strip()
    return rewritten if rewritten else question


def _prepare_answer_prompt(question, chat_history, vectorstore, llm):
    """Runs everything that has to happen BEFORE the final answer can be
    generated: follow-up resolution, sub-question decomposition,
    retrieval, and the relevance guard. None of this is streamed - it's
    all internal groundwork the user doesn't need to watch happen token
    by token, unlike the final answer itself.

    Returns either:
      ("fallback", message_string)   - relevance guard rejected the question
      ("prompt", prompt_text, citation_map) - ready for streamed generation
    """
    standalone_question = rewrite_with_history(question, chat_history, llm)

    decompose_prompt = (
        "Break the following question into 1-3 simple, standalone sub-questions "
        "that together cover everything being asked. If the question is already "
        "simple, just return it as-is. Return ONLY the sub-questions, one per line, "
        "with no numbering or extra text.\n\n"
        f"Question: {standalone_question}"
    )
    decomposition = llm.invoke(decompose_prompt).content
    sub_questions = [q.strip() for q in decomposition.split("\n") if q.strip()]
    if not sub_questions:
        sub_questions = [standalone_question]

    seen_content = {}
    for sub_q in sub_questions:
        sub_results = vectorstore.similarity_search_with_relevance_scores(sub_q, k=3)
        for doc, score in sub_results:
            key = doc.page_content
            if key not in seen_content or score > seen_content[key][1]:
                seen_content[key] = (doc, score)

    if not seen_content or max(score for _, score in seen_content.values()) < RELEVANCE_THRESHOLD:
        return "fallback", build_fallback_message(vectorstore), None

    docs = [doc for doc, _score in seen_content.values()]
    context, citation_map = format_context_with_citations(docs)

    history_text = format_history_for_prompt(chat_history)
    history_block = f"Recent conversation:\n{history_text}\n\n" if history_text else ""

    promptTemplate = ChatPromptTemplate.from_template(
        "You are an expert financial analyst. Answer the question using ONLY the document "
        "context below. You may use the recent conversation for tone and phrasing (e.g. to "
        "answer naturally as a follow-up), but NEVER pull facts, figures, or claims from the "
        "conversation history that aren't backed by the document context.\n\n"
        "{history_block}"
        "The context is split into numbered source blocks, each labeled with its document "
        "and page number.\n\n"
        "Context:\n{context}\n\n"
        "Rules:\n"
        "- Answer as fully as you can using only the context provided, even if it only "
        "partially addresses the question.\n"
        "- Whenever you state a specific fact, figure, or claim, cite the source block it "
        "came from using its bracketed number, e.g. [1]. Cite every fact you use.\n"
        "- If a sentence draws on multiple sources, cite all of them, e.g. [1][2].\n"
        "- If the context is entirely unrelated to the question or contains nothing useful, "
        "say so directly and briefly explain what the context actually covers instead "
        "(e.g. \"The provided documents don't address X; they instead cover Y.\").\n"
        "- Do not use any outside knowledge, even if you know the answer.\n"
        "- Do not invent specific numbers, dates, or facts not present in the context.\n"
        "- Financial documents often contain multiple tables with identically-named rows at "
        "different levels of aggregation (e.g. \"Total net revenue\" or \"Net income\" may appear "
        "once for the whole company AND separately for each business segment, such as Credit "
        "Card, Consumer Banking, or Commercial Banking). Before citing a figure, check which "
        "table and aggregation level it comes from. If a figure is segment-specific rather than "
        "company-wide, say so explicitly (e.g. \"Credit Card segment net revenue was...\") rather "
        "than presenting it as a consolidated total.\n\n"
        "Question: {question}\n"
        "Answer:"
    )

    prompt = promptTemplate.format(
        history_block=history_block,
        context=context,
        question=standalone_question,
    )
    return "prompt", prompt, citation_map


def load_vectorstore_and_llm():
    try:
        embeddings = OpenAIEmbeddings()
    except ImportError as e:
        if "socksio" in str(e) or "SOCKS" in str(e):
            raise ImportError(
                "Missing dependency for SOCKS proxy support. Install with: `pip install httpx[socks]`"
            ) from e
        raise
    except Exception as e:
        if not os.getenv("OPENAI_API_KEY"):
            raise RuntimeError(
                "OPENAI_API_KEY is not set in the environment. Set it and retry."
            ) from e
        raise

    vectorstore = Chroma(
        persist_directory="vectorstore/",
        embedding_function=embeddings
    )
    llm = ChatOpenAI(model="gpt-3.5-turbo")
    return vectorstore, llm


def prepare_answer_prompt(question, chat_history, vectorstore, llm):
    """Public alias - see _prepare_answer_prompt for the actual logic.
    Exposed separately (not prefixed with _) so callers like the
    Streamlit app can run prep under a spinner and stream only the
    generation step that follows."""
    return _prepare_answer_prompt(question, chat_history, vectorstore, llm)


def stream_answer_tokens(llm, prompt):
    """Yields ONLY the answer tokens themselves, with no sources footer -
    used when a caller wants to show prep-phase feedback (e.g. a spinner)
    separately from the token-by-token generation that follows."""
    for chunk in llm.stream(prompt):
        token = chunk.content or ""
        if token:
            yield token


def stream_query_documents(question, chat_history=None):
    """Generator version of query_documents. Yields text chunks as the
    final answer is generated, so a caller (CLI or Streamlit) can display
    it token-by-token instead of waiting for the whole response.

    Everything before the final answer (rewriting, decomposition,
    retrieval, the relevance guard) still runs synchronously first, since
    those steps produce short internal outputs the user doesn't watch
    happen live - only the answer itself benefits from streaming.

    On rejection (relevance guard), yields the fallback message as a
    single chunk. On success, yields answer tokens as they arrive from
    the LLM, followed by one final chunk containing the sources footer.
    """
    chat_history = chat_history or []
    vectorstore, llm = load_vectorstore_and_llm()

    kind, payload, citation_map = _prepare_answer_prompt(question, chat_history, vectorstore, llm)

    if kind == "fallback":
        yield payload
        return

    prompt = payload
    answer_so_far = ""
    for token in stream_answer_tokens(llm, prompt):
        answer_so_far += token
        yield token

    sources_footer = format_sources_footer(citation_map, answer_so_far)
    if sources_footer:
        yield sources_footer


def query_documents(question, chat_history=None):
    """Non-streaming wrapper kept for callers that just want the final
    string in one shot (e.g. tests, scripts). CLI and Streamlit use
    stream_query_documents() directly for the live token-by-token UX."""
    return "".join(stream_query_documents(question, chat_history=chat_history))


if __name__ == "__main__":
    history = []
    print("Ask a question (type 'quit' or 'exit' to stop):\n")
    while True:
        question = input("Ask a question: ").strip()
        if question.lower() in ("quit", "exit"):
            break
        if not question:
            continue

        print("\nAnswer: ", end="", flush=True)
        full_answer = ""
        for chunk in stream_query_documents(question, chat_history=history):
            print(chunk, end="", flush=True)
            full_answer += chunk
        print("\n")

        history.append((question, full_answer))