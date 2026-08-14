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

    # Step 2 - Retrieve with relevance scores so we can guard against
    # off-topic questions before ever calling the LLM
    results = vectorstore.similarity_search_with_relevance_scores(question, k=3)
    for doc, score in results:
        print(f"Score: {score:.3f} | {doc.page_content[:100]}")
    if not results or max(score for _, score in results) < RELEVANCE_THRESHOLD:
        return FALLBACK_ANSWER

    docs = [doc for doc, _score in results]
    context = "\n\n".join(doc.page_content for doc in docs)

    # Step 3 - Initialize the LLM
    llm = ChatOpenAI(model="gpt-3.5-turbo")

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