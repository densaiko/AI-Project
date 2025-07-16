# rag_app_v2.py

import os
from dotenv import load_dotenv
from langchain.text_splitter import CharacterTextSplitter
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.document_loaders import DirectoryLoader, PyPDFLoader
from langchain_community.vectorstores import FAISS

# New style imports
from langchain_core.prompts import ChatPromptTemplate
from langchain.memory import ConversationBufferMemory
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_retrieval_chain

load_dotenv()
os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY", "your-key")

MODEL = "gpt-4o-mini"
VECTORSTORE_DIR = "faiss_index"  # adjust path if your faiss_index is elsewhere

def build_conversational_chain():
    embeddings = OpenAIEmbeddings()

    # Check if saved vectorstore exists
    if os.path.exists(VECTORSTORE_DIR):
        vectorstore = FAISS.load_local(
            VECTORSTORE_DIR, embeddings, allow_dangerous_deserialization=True
        )
    else:
        raise FileNotFoundError(
            f"Vectorstore directory '{VECTORSTORE_DIR}' not found! Upload vectorstore (faiss_index/) to root project."
        )

    retriever = vectorstore.as_retriever()

    # PROMPT: Accepts context, chat_history, and input
    prompt = ChatPromptTemplate.from_messages([
        ("system", "You are a helpful assistant."),
        ("human", "{chat_history}\n\nContext:\n{context}\n\nUser: {input}")
    ])

    # MEMORY: Stores chat history as string for prompt
    memory = ConversationBufferMemory(
        memory_key="chat_history",
        input_key="input",
        return_messages=False
    )

    llm = ChatOpenAI(model=MODEL, temperature=0.7)

    # Combine LLM and prompt with the RAG doc chain
    document_chain = create_stuff_documents_chain(llm, prompt)
    retrieval_chain = create_retrieval_chain(retriever, document_chain)

    # Return both chain and memory, as both are needed in main app/chat loop!
    return retrieval_chain, memory
