import streamlit as st
from query import (
    load_vectorstore_and_llm,
    prepare_answer_prompt,
    stream_answer_tokens,
    format_sources_footer,
)

st.set_page_config(page_title="Fine Print", page_icon="📊")

st.title("📊 Fine Print")
st.caption("Ask questions about the documents in your `data/` folder. Follow-up questions are supported.")

if "history" not in st.session_state:
    st.session_state.history = []


def split_answer_and_sources(full_answer):
    """Split the answer text from its '\\nSources:\\n  [n] ...' footer so
    sources can be rendered as a proper Markdown list."""
    if "\nSources:" not in full_answer:
        return full_answer, []
    answer_part, sources_part = full_answer.split("\nSources:", 1)
    source_lines = [
        line.strip().lstrip("[").replace("]", ":", 1)
        for line in sources_part.strip().split("\n") if line.strip()
    ]
    return answer_part.strip(), source_lines


def render_sources(sources):
    if sources:
        st.markdown("**Sources:**")
        for line in sources:
            st.markdown(f"- {line}")


def render_answer(full_answer):
    """Render a completed (already-streamed) answer, used for history."""
    answer_text, sources = split_answer_and_sources(full_answer)
    st.write(answer_text)
    render_sources(sources)


question = st.text_input("Ask a question", placeholder="e.g. What was net income this quarter?")
ask_clicked = st.button("Ask", type="primary")

if ask_clicked and question.strip():
    st.subheader("Answer")

    # Prep (follow-up rewriting, question decomposition, retrieval, the
    # relevance guard) happens synchronously and isn't itself streamed -
    # it's short internal work, not something worth showing token by
    # token. Running it under a spinner keeps the UI from looking frozen
    # during this part, since without it there's a silent pause before
    # generation starts. Only the actual answer generation that follows
    # is streamed live via st.write_stream.
    with st.spinner("Searching documents..."):
        vectorstore, llm = load_vectorstore_and_llm()
        kind, payload, citation_map = prepare_answer_prompt(
            question, st.session_state.history, vectorstore, llm
        )

    try:
        if kind == "fallback":
            st.write(payload)
            full_answer = payload
        else:
            answer_text = st.write_stream(stream_answer_tokens(llm, payload))
            sources_footer = format_sources_footer(citation_map, answer_text)
            render_sources(
                split_answer_and_sources(answer_text + sources_footer)[1]
            )
            full_answer = answer_text + sources_footer

        st.session_state.history.append((question, full_answer))
    except Exception as e:
        st.error(f"Something went wrong: {e}")

elif st.session_state.history:
    latest_q, latest_a = st.session_state.history[-1]
    st.subheader("Answer")
    render_answer(latest_a)

if len(st.session_state.history) > 1:
    prior = st.session_state.history[:-1]
    st.divider()
    st.subheader("Previous questions")
    for q, a in reversed(prior):
        with st.expander(q):
            render_answer(a)

if st.session_state.history and st.button("Clear conversation"):
    st.session_state.history = []
    st.rerun()

if not st.session_state.history:
    st.info("Ask a question above to get started.")