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

# To mantain the conversation and history
from langchain.chains import create_retrieval_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.messages import HumanMessage, AIMessage
from langchain.chains.combine_documents import create_stuff_documents_chain


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
            extracted_texts.append(None)
    st.success(f"Se han cargado {len(extracted_texts)} páginas")
    return extracted_texts

def get_chunks(raw_text):
    """the text is split into chunks of 1000 characters each."""
    text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=100, separator="\n")
    chunks = text_splitter.split_documents(raw_text)
    return chunks

@st.cache_resource
def get_vector_store(_chunks):
    """Get vectors for each chunk."""
    embeddings = OllamaEmbeddings(model='nomic-embed-text:latest')
    vector_store = FAISS.from_documents(_chunks, embeddings)
    return vector_store

def get_response(query, context):
    """Get a conversation prompt and response."""
    llm = Ollama(model='gemma:2b')
    prompt = ChatPromptTemplate.from_template("""
    You are a helpful assistant.
    Answer the following questions considering the history of the conversation:

    Context: {context}

    Question: {user_question}""")
    document_chain = create_stuff_documents_chain(llm, prompt)
    retrieval_chain = create_retrieval_chain(context.as_retriever(), document_chain)
    return retrieval_chain.stream(
        {"context": context, "user_queston": query})

def main():
    st.set_page_config(page_title="Chatbot", page_icon=":books:")

    st.title("Chatbot")

    # Load PDFs and create the vector store
    with st.sidebar:
        st.subheader("Cargue PDFs")
        pdf_docs = st.file_uploader("Cargar PDF", type=["pdf"], accept_multiple_files=True)
        if st.button("Procesar PDF"):
            with st.spinner("Procesando PDF"):
                if len(pdf_docs) > 0:
                    raw_text = load_pdf(pdf_docs)
                    chunks = get_chunks(raw_text)
                    vector_store = get_vector_store(chunks)

                    if "processed" not in st.session_state:
                        st.session_state.processed = {}
                    st.session_state.processed["vector_store"] = vector_store
                    st.success("Se ha creado la base de datos")
                elif len(pdf_docs) == 0:
                    # Handle case where no PDFs are selected but processed data exists
                    if "processed" in st.session_state and "vector_store" in st.session_state.processed:
                        st.success("Se ha guardado la base de datos (utilizando datos previos)")
                    else:
                        st.error("No se ha seleccionado ningún archivo")
                else:
                    st.error("No se ha seleccionado ningún archivo PDF")
                
    # Chat history
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []

    # Conversation
    for message in st.session_state.chat_history:
        if isinstance(message, HumanMessage):
            with st.chat_message("Human"):
                st.markdown(message.content)
        else:
            with st.chat_message("Ai"):
                st.markdown(message.content)
    
    # User input  
    user_input = st.chat_input("Escriba su pregunta")
    if user_input is not None and user_input != "":
        st.session_state.chat_history.append(HumanMessage(user_input))
        
        with st.chat_message("Human"):
            st.markdown(user_input)
        
        with st.chat_message("Ai"):
            ai_response =  \
            st.write_stream(get_response( \
                user_input, \
                st.session_state.processed["vector_store"]))
        
        st.session_state.chat_history.append(AIMessage(ai_response))

    
    

if __name__ == "__main__":
    main()
