import streamlit as st
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.document_loaders import PyPDFLoader
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from langchain_core.output_parsers import StrOutputParser
import tempfile
import os

# Load environment variables
load_dotenv()

parser = StrOutputParser()

# Streamlit Page Config
st.set_page_config(page_title="📄 Research Agent with Groq", layout="centered")

# Header
st.markdown("""
    <h1 style='text-align: center;'>📚 LLM-Powered Research Agent</h1>
    <p style='text-align: center; color: gray;'>Powered by Groq | Summarize + Chat with Research Papers</p>
""", unsafe_allow_html=True)

# Sidebar
st.sidebar.title("⚙️ Settings")
model_name = st.sidebar.selectbox("Choose Groq Model", ["deepseek-r1-distill-llama-70b", "llama-3.1-8b-instant"])
top_k = st.sidebar.slider("Top-k Chunks to Retrieve", 1, 10, 3)

# Upload
st.markdown("### 📤 Upload a Research Paper")
pdf_file = st.file_uploader("Upload PDF", type=["pdf"])

# State for storing vectorstore, retriever, and memory
if "retriever" not in st.session_state:
    st.session_state.retriever = None
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "summary" not in st.session_state:
    st.session_state.summary = None

if pdf_file:
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
        tmp.write(pdf_file.read())
        pdf_path = tmp.name

    with st.spinner("📖 Reading and processing document..."):
        loader = PyPDFLoader(pdf_path)
        pages = loader.load()
        splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        docs = splitter.split_documents(pages)

        # Embeddings & Vector Store
        embeddings = HuggingFaceEmbeddings(model_name="BAAI/bge-small-en")
        vectorstore = FAISS.from_documents(docs, embeddings)
        st.session_state.retriever = vectorstore.as_retriever(search_type="similarity", k=top_k)

        # Groq LLM
        llm = ChatGroq(model=model_name, temperature=0.1)

        # Generate Summary
        summary_prompt = PromptTemplate(
            input_variables=["context"],
            template=(
                "You are an academic summarizer.\n"
                "Given the following text, write a concise and formal research paper summary. "
                "Do not explain your thoughts or process, just return the final summary only.\n\n"
                "{context}"
            )
        )

        summary_chain = summary_prompt | llm |parser
        summary_text = pages[0].page_content[:3000]  # Short context for summary
        summary_result = summary_chain.invoke({"context": summary_text})
        st.session_state.summary = summary_result

    st.success("✅ Document processed successfully!")
    os.remove(pdf_path)

if st.session_state.summary:
    st.markdown("---")
    st.markdown("### 📝 Paper Summary")
    st.info(st.session_state.summary)

# Chat Interface
st.markdown("---")
st.markdown("### 💬 Chat with the Paper")
query = st.text_input("Ask a question about the paper:")

if query and st.session_state.retriever:
    llm = ChatGroq(model=model_name, temperature=0.1)
    memory = ConversationBufferMemory(
    memory_key="chat_history",
    return_messages=True,
    output_key="answer"  # 👈 this line fixes the ValueError
    )

    qa_chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=st.session_state.retriever,
        memory=memory,
        return_source_documents=True
    )

    with st.spinner("🤖 Generating Answer..."):
        result = qa_chain({"question": query, "chat_history": st.session_state.chat_history})
        answer = result["answer"]
        st.session_state.chat_history.append((query, answer))

    st.markdown("#### 🧠 Answer")
    st.success(answer)

    with st.expander("🔍 Retrieved Chunks"):
        for i, doc in enumerate(result["source_documents"]):
            st.markdown(f"**Chunk {i+1}:**")
            st.info(doc.page_content.strip())

# Show chat history
if st.session_state.chat_history:
    st.markdown("---")
    st.markdown("### 💬 Chat History")
    for i, (q, a) in enumerate(st.session_state.chat_history[::-1]):
        st.markdown(f"**Q{i+1}: {q}**")
        st.markdown(f"*A{i+1}: {a}*")