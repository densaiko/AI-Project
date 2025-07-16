import streamlit as st
from html import escape
from rag_app_v2 import build_conversational_chain
import streamlit.components.v1 as components

st.set_page_config(page_title="RAG Chatbot", layout="wide")

# === Sidebar ===
with st.sidebar:
    st.image("AHM.png", width=200)
    st.markdown("## 📚 RAG AI Assistant")
    st.markdown("A smart assistant powered by LLM + vector database (RAG).")
    st.markdown("---")
    st.markdown("### ℹ️ How to Use")
    st.markdown("""
    1. Load your document-based vector index.
    2. Ask a question about the document.
    3. Chatbot will retrieve & generate the answer.
    """)
    st.markdown("---")
    st.markdown("Made with ❤️ using Streamlit + LangChain")

# === Load RAG Chain & Memory ===
@st.cache_resource
def load_chain():
    return build_conversational_chain()

retrieval_chain, memory = load_chain()

# === Session state init ===
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

if "scroll_to_bottom" not in st.session_state:
    st.session_state.scroll_to_bottom = False

# === CSS for chat bubble styling ===
st.markdown("""
<style>
.chat-bubble {
    padding: 10px 15px;
    border-radius: 15px;
    font-size: 15px;
    line-height: 1.5;
    max-width: 75%;
    margin-bottom: 10px;
    word-wrap: break-word;
    box-shadow: 0 1px 3px rgba(0,0,0,0.1);
}
.user {
    background-color: #d4fcdc;
    align-self: flex-end;
    margin-left: auto;
    border-bottom-right-radius: 0;
}
.bot {
    background-color: #ffffff;
    align-self: flex-start;
    margin-right: auto;
    border-bottom-left-radius: 0;
}
</style>
""", unsafe_allow_html=True)

# === Title & Form Input ===
st.markdown("<h1 style='text-align: center; color: #00BFFF;'>🧠 RAG Chatbot with LLM + Vector Search</h1>", unsafe_allow_html=True)
st.markdown("### 💬 Ask a question about the document:")

with st.form(key="chat_form"):
    user_input = st.text_input("Kamu:", placeholder="Tulis pertanyaanmu di sini...", key="input_text")
    submitted = st.form_submit_button("Kirim")

# === Process Chat ===
if submitted and user_input:
    with st.spinner("Memikirkan jawaban..."):
        # --- Format chat history for the chain prompt (merge user/assistant pairs) ---
        chat_history_pairs = [
            f"User: {u}\nAssistant: {a}"
            for u, a in zip(
                [msg for idx, (sender, msg) in enumerate(st.session_state.chat_history) if sender == "Kamu"],
                [msg for idx, (sender, msg) in enumerate(st.session_state.chat_history) if sender == "Astra Honda Assistant"]
            )
        ]
        # If last turn is a user question without answer yet
        if len(st.session_state.chat_history) % 2 != 0 and st.session_state.chat_history[-1][0] == "Kamu":
            last_user = st.session_state.chat_history[-1][1]
            chat_history_pairs.append(f"User: {last_user}")

        chat_history_str = "\n".join(chat_history_pairs)
        
        # --- Modern RAG chain call ---
        result = retrieval_chain.invoke({
            "input": user_input,
            "chat_history": chat_history_str,
        })
        answer = result["answer"]

        # Save chat to session and update memory
        st.session_state.chat_history.append(("Kamu", user_input))
        st.session_state.chat_history.append(("Astra Honda Assistant", answer))
        memory.save_context({"input": user_input}, {"output": answer})

        # Trigger scroll
        st.session_state.scroll_to_bottom = True

# === Display Chat History ===
st.markdown("### 🧾 Conversation")

for sender, message in st.session_state.chat_history:
    safe_msg = escape(message).replace("\n", "<br>").replace("`", "'")
    role_class = "user" if sender == "Kamu" else "bot"
    st.markdown(f"""
        <div class="chat-bubble {role_class}">
            <b>{sender}:</b><br>{safe_msg}
        </div>
    """, unsafe_allow_html=True)

# === Auto-scroll to bottom after message ===
if st.session_state.get("scroll_to_bottom", False):
    components.html(
        """
        <script>
            window.scrollTo({ top: document.body.scrollHeight, behavior: "smooth" });
        </script>
        """,
        height=0,
    )
    st.session_state.scroll_to_bottom = False
