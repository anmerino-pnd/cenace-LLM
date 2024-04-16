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


def load_pdf(file):
    """Load a PDF file and split it into pages."""
    with open(file.name, "rb") as f:
        pdf_loader = pdf_loader.PyPDFLoader(f)
        page = pdf_loader.load_and_split()
        return page

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
        pdf_docs = st.file_uploader("Cargar PDF", type=["pdf"])
        if st.button("Procesar PDF"):
            with st.spinner("Procesando PDF"):
                # get pdf text
                if pdf_docs is not None:
                    raw_text = load_pdf(pdf_docs)
                else:
                    st.error("No se ha seleccionado ningún archivo PDF")
                # get chunks
                
                # get vectors



if __name__ == "__main__":
    main()
