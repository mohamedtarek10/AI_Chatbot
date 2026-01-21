import uuid
import streamlit as st

from main import init_rag_system, get_answer


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

    col1, col2 = st.columns([3, 1])
    with col2:
        if st.button("Start new session", use_container_width=True):
            st.session_state.session_id = str(uuid.uuid4())
            st.session_state.messages = []
            st.toast("New session created.")

 
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
                    answer = f"Something went wrong: {exc}"
                    sources = None
                st.markdown(answer)

               
        append_message("assistant", answer)


if __name__ == "__main__":
    main()
