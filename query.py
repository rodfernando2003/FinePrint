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
    standalone, this is a no-op (the LLM is instructed to return it as-is)."""
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


def query_documents(question, chat_history=None):
    """chat_history is an optional list of (question, answer_text) tuples
    from earlier turns in the same conversation, oldest first. Pass None
    or [] for a fresh, single-turn query (same behavior as before)."""
    chat_history = chat_history or []

    # Step 1 - Load the existing vectorstore
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

    # Step 1b - Resolve follow-up references ("that", "it", "compared to
    # last quarter") into a standalone question before doing anything else.
    # Everything downstream (decomposition, retrieval, citations) uses
    # this resolved version.
    standalone_question = rewrite_with_history(question, chat_history, llm)

    # Step 2a - Decompose the (now standalone) question into sub-questions.
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

    # Step 2b - Retrieve for each sub-question separately, then merge and
    # dedupe. Track the best score per unique chunk for the relevance guard.
    seen_content = {}
    for sub_q in sub_questions:
        sub_results = vectorstore.similarity_search_with_relevance_scores(sub_q, k=3)
        for doc, score in sub_results:
            key = doc.page_content
            if key not in seen_content or score > seen_content[key][1]:
                seen_content[key] = (doc, score)

    if not seen_content or max(score for _, score in seen_content.values()) < RELEVANCE_THRESHOLD:
        return build_fallback_message(vectorstore)

    docs = [doc for doc, _score in seen_content.values()]
    context, citation_map = format_context_with_citations(docs)

    # Step 3 - Prompt includes recent conversation history so the final
    # answer can be phrased naturally as a follow-up (e.g. "It was $2.1
    # billion" instead of re-stating the full subject every time), while
    # still being restricted to only the retrieved document context for
    # any actual facts/figures.
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
        "- Do not invent specific numbers, dates, or facts not present in the context.\n\n"
        "Question: {question}\n"
        "Answer:"
    )

    prompt = promptTemplate.format(
        history_block=history_block,
        context=context,
        question=standalone_question,
    )
    response = llm.invoke(prompt)
    answer_text = response.content

    sources_footer = format_sources_footer(citation_map, answer_text)

    return answer_text + sources_footer


if __name__ == "__main__":
    history = []
    print("Ask a question (type 'quit' or 'exit' to stop):\n")
    while True:
        question = input("Ask a question: ").strip()
        if question.lower() in ("quit", "exit"):
            break
        if not question:
            continue
        answer = query_documents(question, chat_history=history)
        print(f"\nAnswer: {answer}\n")
        history.append((question, answer))