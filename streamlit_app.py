import os
import streamlit as st


# To create the vector store, we need to load the PDF file
# split it into pages, split the pages into chunks
# and get the vectors for each chunk.
from langchain_community.vectorstores import FAISS
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.document_loaders import PyPDFLoader


# To have a Chat prompt and response
from langchain_community.llms import Ollama
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

# To create the retrieval chain that will look for the answer
# in the vector store.
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain


def load_pdf(pdf_files):
    """Loads a list of PDF files and combines their pages into a single list.

    Args:
        pdf_files (list): A list of file paths or Streamlit UploadedFile objects.

    Returns:
        list: A list containing all pages from all the PDFs.
    """

    combined_pages = []
    for file in pdf_files:
        try:
            # Handle both file paths and Streamlit UploadedFile objects
            if isinstance(file, str):
                    pdf_loader = PyPDFLoader(f)
                    pages = pdf_loader.load_and_split()
            else:
                    pdf_loader = PyPDFLoader(f)
                    pages = pdf_loader.load_and_split()

            # Extract and concatenate page text directly
            for page in pages:
                combined_text += str(page) + " "  # Convert page to string and add space
        except Exception as e:
            st.error(f"Error processing file {file}: {e}")  # Informative error handling

    return combined_pages

# def get_chunks(page):
#     """the text is split into chunks of 1000 characters each."""
#     text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
#     chunks = text_splitter.split_documents(page)
#     return chunks

# def get_vector_store(chunks):
#     """Get vectors for each chunk."""
#     embeddings = OllamaEmbeddings(model='nomic-embed-text:latest') 
#     vector_store = FAISS.from_documents(chunks, embeddings)
#     vector_store.save_local("Character_FAISS_nomic")

# def get_conversational_chain():
#     """Get a conversation prompt and response."""
#     llm = Ollama(model='gemma:2b')
#     chat_prompt = ChatPromptTemplate.from_template("""Answer the following question based only on the provided context:

# <context>
# {context}
# </context>

# Question: {input}""")
#     chat_output_parser = StrOutputParser()
#     document_chain = create_stuff_documents_chain(llm, chat_prompt, chat_output_parser)
#     return document_chain

# def user_input(user_question):
#     """Get user input and return the response."""
#     embeddings = OllamaEmbeddings(model='nomic-embed-text:latest')
#     new_vector_store = FAISS.load_local("Character_FAISS_nomic", embeddings)
#     chain = get_conversational_chain()
#     retriever = new_vector_store.as_retriever()
#     retrieval_chain = create_retrieval_chain(retriever, chain)
#     st.write(retrieval_chain.invoke({"input": user_question}))

def main():
    st.set_page_config(page_title="Chatbot", page_icon=":books:")
    st.header("Chatbot")
    st.text_input("Haga una pregunta sobre sus documentos")

    with st.sidebar:
        st.subheader("Cargue PDFs")
        pdf_docs = st.file_uploader("Cargar PDF", type=["pdf"], accept_multiple_files=True)
        if st.button("Procesar PDF"):
            with st.spinner("Procesando PDF"):
                if pdf_docs is not None:
                    for file in pdf_docs:
                        # Process each PDF
                        try:
                            st.success(f"Procesado: {file.name}")
                            # Load the PDF and combine its pages
                            pages = load_pdf([file])
                            st.success(f"Se han cargado {len(pages)} páginas")
                            # Split the pages into chunks
                            # chunks = get_chunks(pages)

                        except Exception as e:
                            st.error(f"Error procesando {file.name}: {e}")
                else:
                    st.error("No se ha seleccionado ningún archivo PDF")



if __name__ == "__main__":
    main()
