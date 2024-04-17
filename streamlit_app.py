import os
import tempfile
import streamlit as st
# Model we will use to interact with the vector store
from langchain_community.llms import Ollama

# To create the vector store, we need to load the PDF file
# split it into pages, split the pages into chunks
# and get the vectors for each chunk.
from langchain_community.vectorstores.faiss import FAISS
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.document_loaders import PyPDFLoader

# To create the retrieval chain that will look for the answer
# in the vector store.
from langchain.chains import ConversationalRetrievalChain

# To mantain the conversation and history
from langchain.memory import ConversationBufferMemory

# html design of the chatbot interface
from htmlTemplates import css, bot_template, user_template


def load_pdf(pdf_files):
    extracted_texts = []
    for pdf in pdf_files:
        try:
            # Create a temporary file for each PDF
            with tempfile.NamedTemporaryFile(delete=False) as temp_file:
                temp_file.write(pdf.read())
                file_path = temp_file.name

            # Use PyPDFLoader to process the PDF
            pdf_loader = PyPDFLoader(file_path)
            pages = pdf_loader.load_and_split()
            extracted_texts += pages

        except Exception as e:
            print(f"Error procesando archivo {pdf.name}: {e}")
    st.success(f"Se han cargado {len(extracted_texts)} páginas")
    return extracted_texts

def get_chunks(raw_text):
    """the text is split into chunks of 1000 characters each."""
    text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=100, separator="\n")
    chunks = text_splitter.split_documents(raw_text)
    return chunks

def get_vector_store(chunks):
    """Get vectors for each chunk."""
    embeddings = OllamaEmbeddings(model='nomic-embed-text:latest')
    vector_store = FAISS.from_documents(chunks, embeddings)
    return vector_store

def get_conversational_chain(VectorStore):
    """Get a conversation prompt and response."""
    llm = Ollama(model='tinyllama:latest')
    memory = ConversationBufferMemory(memory_key = 'chat_history', return_messages= True)
    conversation_chain = ConversationalRetrievalChain.from_llm(
        llm = llm,
        retriever= VectorStore.as_retriever(),
        memory = memory
    )
    return conversation_chain

def main():
    st.set_page_config(page_title="Chatbot", page_icon=":books:")
    st.write(css, unsafe_allow_html=True)

    st.header("Chatbot")

    with st.sidebar:
        st.subheader("Cargue PDFs")
        pdf_docs = st.file_uploader("Cargar PDF", type=["pdf"], accept_multiple_files=True)
        if st.button("Procesar PDF"):
            with st.spinner("Procesando PDF"):
                if pdf_docs is not None:
                    raw_text = load_pdf(pdf_docs)
                    chunks = get_chunks(raw_text)
                    vectore_store = get_vector_store(chunks)
                    st.success("Se ha creado la base de datos")
                    if "processed" not in st.session_state:
                        st.session_state.processed = {
                            "chunks": chunks,
                            "vector_store": vectore_store
                        
                        }
                    else:
                        st.session_state.processed["chunks"] = chunks
                        st.session_state.processed["vector_store"] = vectore_store
                else:
                    st.error("No se ha seleccionado ningún archivo PDF")
    
    st.session_state.conversation = get_conversational_chain(
                        vectore_store)

    if "messages" not in st.session_state:
        st.session_state.messages = []
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    if user_question := st.chat_input("Haga una pregunta sobre sus documentos")
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(user_question)

        history = [
            f"{message['role']}: {message['content']}" 
            for message in st.session_state.messages
        ]
    
        result = st.session_state.conversation({
            "question": user_question, 
            "chat_history": history
        })

        with st.chat_message("assistant"):
            message_placeholder = st.empty()
            full_response = result["answer"]
            message_placeholder.markdown(full_response + "|")
        message_placeholder.markdown(full_response)    
        print(full_response)
        st.session_state.messages.append({"role": "assistant", "content": full_response})

if __name__ == "__main__":
    main()
