from dotenv import load_dotenv
import os

# Your imports go here - you need:
# 1. OpenAIEmbeddings from langchain_openai
# 2. ChatOpenAI from langchain_openai
# 3. Chroma from langchain_community.vectorstores
# 4. ChatPromptTemplate from langchain.prompts
# 5. StrOutputParser from langchain_core.output_parsers
# 6. RunnablePassthrough from langchain_core.runnables
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_chroma import Chroma
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough, RunnableMap    

load_dotenv()

def query_documents(question):

    # Step 1 - Load the existing vectorstore
    # Provide clearer errors when environment or dependencies are misconfigured
    try:
        embeddings = OpenAIEmbeddings()
    except ImportError as e:
        if "socksio" in str(e) or "SOCKS" in str(e):
            raise ImportError(
                "Missing dependency for SOCKS proxy support. Install with: `pip install httpx[socks]`"
            ) from e
        raise
    except Exception as e:
        # Common root causes: missing OPENAI_API_KEY or proxy misconfiguration
        if not os.getenv("OPENAI_API_KEY"):
            raise RuntimeError(
                "OPENAI_API_KEY is not set in the environment. Set it and retry."
            ) from e
        raise
    vectorstore = Chroma(
        persist_directory="vectorstore/",
        embedding_function=embeddings
    )

    # Step 2 - Create a retriever from the vectorstore
    retriever = vectorstore.as_retriever(search_kwargs={"k": 3})

    # Step 3 - Initialize the LLM
    llm = ChatOpenAI(model="gpt-3.5-turbo")

    # Step 4 - Create a prompt template
    promptTemplate = ChatPromptTemplate.from_template(
        "You are an expert financial analyst that answers questions exclusively based on the following context: {context}\n\nQuestion: {question}\nAnswer:"
    )

    # Step 5 - Build the chain using LCEL pipe syntax
    def format_docs(docs):
        return "\n\n".join(doc.page_content for doc in docs)

    chain = RunnableMap({
        "context": retriever | format_docs,
        "question": RunnablePassthrough()
        }) | promptTemplate | llm | StrOutputParser()

    # Step 6 - Invoke the chain with the question and return the result
    return chain.invoke(question)
if __name__ == "__main__":
    question = input("Ask a question: ")
    answer = query_documents(question)
    print(f"\nAnswer: {answer}")