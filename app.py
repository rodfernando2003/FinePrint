import streamlit as st
from query import query_documents

st.set_page_config(page_title="Fine Print", page_icon="📊")

st.title("📊 Fine Print")
st.caption("Ask questions about the documents in your `data/` folder. Follow-up questions are supported.")

if "history" not in st.session_state:
    st.session_state.history = []


def split_answer_and_sources(full_answer):
    """query_documents() returns the answer with a '\\nSources:\\n  [n] ...'
    footer appended as one string. Split them apart so sources can be
    rendered as a proper Markdown list - st.write() collapses single
    newlines, so left as one blob the citations would run together."""
    if "\nSources:" not in full_answer:
        return full_answer, []
    answer_part, sources_part = full_answer.split("\nSources:", 1)
    source_lines = [
        line.strip().lstrip("[").replace("]", ":", 1)
        for line in sources_part.strip().split("\n") if line.strip()
    ]
    return answer_part.strip(), source_lines


def render_answer(full_answer):
    answer_text, sources = split_answer_and_sources(full_answer)
    st.write(answer_text)
    if sources:
        st.markdown("**Sources:**")
        for line in sources:
            st.markdown(f"- {line}")


question = st.text_input("Ask a question", placeholder="e.g. What was net income this quarter?")
ask_clicked = st.button("Ask", type="primary")

if ask_clicked and question.strip():
    with st.spinner("Searching documents and generating answer..."):
        try:
            answer = query_documents(question, chat_history=st.session_state.history)
            st.session_state.history.append((question, answer))
        except Exception as e:
            st.error(f"Something went wrong: {e}")

if st.session_state.history:
    latest_q, latest_a = st.session_state.history[-1]
    st.subheader("Answer")
    render_answer(latest_a)

    if len(st.session_state.history) > 1:
        st.divider()
        st.subheader("Previous questions")
        for q, a in reversed(st.session_state.history[:-1]):
            with st.expander(q):
                render_answer(a)

    if st.button("Clear conversation"):
        st.session_state.history = []
        st.rerun()
else:
    st.info("Ask a question above to get started.")