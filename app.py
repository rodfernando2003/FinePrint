import streamlit as st
from query import query_documents

st.set_page_config(page_title="Financial Services RAG", page_icon="📊")

st.title("📊 Financial Services RAG")
st.caption("Ask questions about Capital One's Financials")

# Keep a simple chat-style history in session state
if "history" not in st.session_state:
    st.session_state.history = []

question = st.text_input("Ask a question", placeholder="e.g. What was net income this quarter?")
ask_clicked = st.button("Ask", type="primary")

if ask_clicked and question.strip():
    with st.spinner("Searching documents and generating answer..."):
        try:
            answer = query_documents(question)
            st.session_state.history.append((question, answer))
        except Exception as e:
            st.error(f"Something went wrong: {e}")

# Show most recent answer prominently
if st.session_state.history:
    latest_q, latest_a = st.session_state.history[-1]
    st.subheader("Answer")
    st.write(latest_a)

    # Show prior Q&A pairs, most recent first
    if len(st.session_state.history) > 1:
        st.divider()
        st.subheader("Previous questions")
        for q, a in reversed(st.session_state.history[:-1]):
            with st.expander(q):
                st.write(a)
else:
    st.info("Ask a question above to get started.")