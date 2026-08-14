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

FALLBACK_ANSWER = "I don't have enough information in the provided documents to answer that."


def query_documents(question):

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

    # Step 2 - Initialize the LLM (needed for decomposition + final answer)
    llm = ChatOpenAI(model="gpt-3.5-turbo")

    # Step 2a - Decompose the question into sub-questions. A single compound
    # question (e.g. "where did income grow AND where did they invest")
    # produces one blended embedding that matches neither topic well.
    # Splitting it lets us retrieve separately for each part.
    decompose_prompt = (
        "Break the following question into 1-3 simple, standalone sub-questions "
        "that together cover everything being asked. If the question is already "
        "simple, just return it as-is. Return ONLY the sub-questions, one per line, "
        "with no numbering or extra text.\n\n"
        f"Question: {question}"
    )
    decomposition = llm.invoke(decompose_prompt).content
    sub_questions = [q.strip() for q in decomposition.split("\n") if q.strip()]
    if not sub_questions:
        sub_questions = [question]

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
        for doc, score in sorted(seen_content.values(), key=lambda x: -x[1])[:5]:
            print(f"Score: {score:.3f} | {doc.page_content[:100]}")
        return FALLBACK_ANSWER
    docs = [doc for doc, _score in seen_content.values()]
    context = "\n\n".join(doc.page_content for doc in docs)

    # Step 4 - Create a prompt template that explicitly forces refusal
    # when context is insufficient, rather than relying on the model's
    # judgment to "stay in scope."
    promptTemplate = ChatPromptTemplate.from_template(
        "You are an expert financial analyst. Answer the question using ONLY the context below.\n\n"
        "Context:\n{context}\n\n"
        "Rules:\n"
        "- If the context does not contain enough information to answer the question, respond exactly with: "
        f"\"{FALLBACK_ANSWER}\"\n"
        "- Do not use any outside knowledge, even if you know the answer.\n"
        "- Do not speculate or guess.\n\n"
        "Question: {question}\n"
        "Answer:"
    )

    # Step 5 - Format the prompt and invoke the LLM directly
    # (no need for the RunnableMap/retriever chain since we already
    # did retrieval manually above to get the relevance scores)
    prompt = promptTemplate.format(context=context, question=question)
    response = llm.invoke(prompt)

    return response.content


if __name__ == "__main__":
    question = input("Ask a question: ")
    answer = query_documents(question)
    print(f"\nAnswer: {answer}")