import uuid
import streamlit as st

from main import init_rag_system, get_answer, add_url_to_knowledge_base


st.set_page_config(
    page_title="EV Charging Chatbot",
    page_icon="⚡",
    layout="wide",
)


def init_session() -> None:
    """Initialize Streamlit session state for chat."""
    if "session_id" not in st.session_state:
        st.session_state.session_id = str(uuid.uuid4())
    if "messages" not in st.session_state:
        st.session_state.messages = []


def render_history() -> None:
    """Render existing chat messages."""
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])


def append_message(role: str, content: str) -> None:
    """Store a new message in chat history."""
    st.session_state.messages.append({"role": role, "content": content})


@st.cache_resource
def get_rag_resources():
    """Initialize and cache the heavy RAG resources."""
    return init_rag_system()


def main() -> None:
    init_session()

    st.title("EV Charging Knowledge Assistant")
    st.caption("Ask questions about the uploaded charging station documents.")

    # Initialize resources
    with st.spinner("Initializing system..."):
        resources = get_rag_resources()

    with st.sidebar:
        st.header("Add Knowledge")
        url_input = st.text_input("Enter URL to scrape")
        if st.button("Add URL"):
            if url_input:
                with st.spinner("Scraping and indexing..."):
                    success, msg = add_url_to_knowledge_base(url_input, resources)
                    if success:
                        st.success(msg)
                        # Clear cache to force reload of the updated vectorstore
                        st.cache_resource.clear()
                        # Rerun to refresh the session with new resources
                        st.rerun()
                    else:
                        st.error(f"Failed: {msg}")

        # st.markdown("---")
        # if st.button("Reset Knowledge Base", type="primary"):
        #     with st.spinner("Resetting..."):
        #         import shutil
        #         import os
        #         if os.path.exists("faiss_ev"):
        #             shutil.rmtree("faiss_ev")
        #         st.cache_resource.clear()
        #         st.success("Knowledge base reset. Please refresh.")
        #         st.rerun()

    col1, col2 = st.columns([3, 1])
    with col2:
        if st.button("Start new session", use_container_width=True):
            st.session_state.session_id = str(uuid.uuid4())
            st.session_state.messages = []
            st.toast("New session created.")

    # with st.expander("How this works"):
    #     st.markdown(
    #         "- Your questions are answered using the indexed PDF documents in `files/`.\n"
    #         "- Each chat session keeps only the last couple of turns for context.\n"
    #         "- Sources include file name and modified date when available."
    #     )

    render_history()

    prompt = st.chat_input("Ask about charging stations...")
    if prompt:
        append_message("user", prompt)
        with st.chat_message("user"):
            st.markdown(prompt)

        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                try:
                    result = get_answer(prompt, st.session_state.session_id, resources)
                    answer = result.get("answer") if isinstance(result, dict) else str(result)
                    sources = result.get("context") if isinstance(result, dict) else None
                except Exception as exc:
                    import traceback
                    traceback.print_exc()
                    answer = f"Something went wrong: {exc}"
                    sources = None
                st.markdown(answer)

                # if sources:
                #     with st.expander("Sources"):
                #         for idx, doc in enumerate(sources, start=1):
                #             meta = doc.metadata or {}
                #             file_name = meta.get("file_name") or meta.get("source") or "unknown"
                #             last_modified = meta.get("last_modified_date", "unknown")
                #             st.markdown(f"**{idx}. {file_name}** — Last modified: {last_modified}")

        append_message("assistant", answer)


if __name__ == "__main__":
    main()
