import os
import tempfile
import streamlit as st
# Model we will use to interact with the vector store
from langchain_community.llms import Ollama

# To create the vector store, we need to load the PDF file
# split it into pages, split the pages into chunks
# and get the vectors for each chunk.
from langchain_community.vectorstores import Qdrant
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
            print(f"Error processing file {pdf.name}: {e}")
    st.success(f"Se han cargado {len(extracted_texts)} páginas")
    return extracted_texts

def get_chunks(raw_text):
    """the text is split into chunks of 1000 characters each."""
    text_splitter = CharacterTextSplitter(separator= "\n" ,chunk_size=1000, chunk_overlap=200)
    chunks = text_splitter.split_documents(raw_text)
    return chunks

def get_vector_store(chunks):
    """Get vectors for each chunk."""
    embeddings = OllamaEmbeddings(model='gemma:2b')
    vector_store = Qdrant.afrom_documents(chunks, embeddings)
    return vector_store

def get_conversational_chain(VectorStore, retriever):
    """Get a conversation prompt and response."""
    llm = Ollama(model='gemma:2b')
    retriever = Qdrant.as_retriever(search_type="mmr")
    memory = ConversationBufferMemory(memory_key = 'chat_history', return_messages= True)
    conversation_chain = ConversationalRetrievalChain.from_llm(
        llm = llm,
        memory = memory
    )
    return conversation_chain

def handle_user_input(user_question):
    response = st.session_state.conversation({'question': user_question})
    st.session_state.chat_history = response['chat_history']
    for i, message in enumerate(st.session_state.chat_history):
        if i % 2 == 0:
            st.write(user_template.replace("{{MSG}}", message.content, unsafe_allow_html=True))
        else:
            st.write(bot_template.replace("{{MSG}}", message.content, unsafe_allow_html=True))

def main():
    st.set_page_config(page_title="Chatbot", page_icon=":books:")
    st.write(css, unsafe_allow_html=True)

    st.header("Chatbot")
    user_question = st.text_input("Haga una pregunta sobre sus documentos")
    if user_question:
        handle_user_input(user_question)

    if "conversation" not in st.session_state:
        st.session_state.conversation = None

    if "chat_history" not in st.session_state:
        st.session_state.chat_history = None
    
    # st.write(user_template, unsafe_allow_html=True)
    # st.write(bot_template, unsafe_allow_html=True)

    with st.sidebar:
        st.subheader("Cargue PDFs")
        pdf_docs = st.file_uploader("Cargar PDF", type=["pdf"], accept_multiple_files=True)
        if st.button("Procesar PDF"):
            with st.spinner("Procesando PDF"):
                if pdf_docs is not None:
                    raw_text = load_pdf(pdf_docs)
                    chunks = get_chunks(raw_text)
                    vectore_store = get_vector_store(chunks)
                    st.session_state.conversation = get_conversational_chain(
                        vectore_store)
                else:
                    st.error("No se ha seleccionado ningún archivo PDF")



if __name__ == "__main__":
    main()
