from __future__ import annotations

import os
import shutil
from pprint import pp

import streamlit as st
from langchain_community.chat_models import ChatOllama
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_retrieval_chain
from langchain import hub
from langchain_core.messages import HumanMessage, AIMessage

import rag

DEFAULT_PROMPT_MODEL = "gemma:2b"
DEFAULT_DATABASE_PATH = os.path.abspath("./cenace-index.json")
LOG_ENABLED = True
LOG_LEVEL = 3

def log(level, *args):
    if LOG_ENABLED and LOG_LEVEL >= level:
        print(*args)

def get_prompt_model():
    return st.session_state.get("prompt_model")

def load_prompt_model(fallback: str):
    if "prompt_model" in st.session_state:
        return st.session_state["prompt_model"]
    log(0, "* Loading prompt model name")
    model = os.environ.get("OLLAMA_MODEL")
    log(1, "Reading OLLAMA_MODEL environment variable, got", model)
    if model is None:
        log(1, "Falling back to", fallback)
        model = fallback
    st.session_state["prompt_model"] = model
    return st.session_state["prompt_model"]

def get_rag_database():
    return st.session_state.get("rag_database")

def load_rag_database():
    path = DEFAULT_DATABASE_PATH
    if "rag_database" in st.session_state:
        return st.session_state["rag_database"]
    log(0, "* Loading RAG database from", path)
    if not os.path.exists(path):
        log(1, "Does not exist, creating empty database")
        rag.Database({}).save(path)
    db = rag.Database.load(path)
    st.session_state["rag_database"] = db
    return st.session_state["rag_database"]

def get_selected_collection():
    return st.session_state.get("selected_collection")

def set_selected_collection(id_coll):
    st.session_state["selected_collection"] = id_coll

def load_selected_collection():
    if "selected_collection" in st.session_state:
        return st.session_state["selected_collection"]
    log(0, "* Unselecting collection")
    st.session_state["selected_collection"] = None
    return st.session_state["selected_collection"]

def get_selected_vectorstore():
    return st.session_state.get("selected_vectorstore")

def set_selected_vectorstore(store):
    st.session_state["selected_vectorstore"] = store

def load_selected_vectorstore():
    if "selected_vectorstore" in st.session_state:
        return st.session_state["selected_vectorstore"]
    log(0, "* Setting selected vectorstore")
    if st.session_state["selected_collection"] is None:
        log(1, "No selection, setting None")
        st.session_state["selected_vectorstore"] = None
    else:
        log(1, "Loading for selected collection", st.session_state["selected_collection"])
        db = get_selected_vectorstore()
        id = get_selected_collection()
        if db is None or id is None:
            log(2, "No selected collection")
            st.session_state["selected_vectorstore"] = None
        else:
            log(2, "Loading vectorstore")
            collection = db.index[id]
            store = collection.load_store()
            st.session_state["selected_vectorstore"] = store
    return st.session_state["selected_vectorstore"]

def load_temperature():
    if "temperature" in st.session_state:
        return st.session_state["temperature"]
    log(0, "* Setting initial temperature")
    st.session_state["temperature"] = 0.5
    return st.session_state["temperature"]

def set_temperature(temp):
    st.session_state["temperature"] = temp

def load_seed():
    if "seed" in st.session_state:
        return st.session_state["seed"]
    log(0, "* Setting initial seed")
    st.session_state["seed"] = -1
    return st.session_state["seed"]

def set_seed(seed):
    st.session_state["seed"] = seed

def log_session_state():
    if not (LOG_ENABLED and LOG_LEVEL >= 3):
        return
    db = get_rag_database()
    pp({
        "prompt_model": get_prompt_model(),
        "rag_database": db if db is None else { id: coll.to_dict() for id, coll in db.index.items() },
        "selected_collection": get_selected_collection(),
        "selected_vectorstore": get_selected_vectorstore(),
    }, compact=True)
        

def handle_select_collection(coll_id):
    with st.spinner(text="Cargando colección..."):
        set_selected_collection(coll_id)
        db = get_rag_database()
        assert db is not None
        store = db.index[coll_id].load_store()
        set_selected_vectorstore(store)

def handle_create_collection():
    name = st.session_state["new-coll-name"].strip()
    if name == "":
        st.toast("Falta el nombre de la colección", icon="⚠️")
        return
    description = st.session_state["new-coll-description"].strip()
    if description == "":
        st.toast("Falta la descripción de la colección", icon="⚠️")
        return
    embeddings_model = st.session_state["new-coll-embeddings-model"]
    db = get_rag_database()
    assert db is not None
    coll = rag.Collection.empty(name, description, embeddings_model)
    db.index[str(coll.id)] = coll
    db.save(DEFAULT_DATABASE_PATH)
    st.session_state["new-coll-name"] = ""
    st.session_state["new-coll-description"] = ""
    st.session_state["new-coll-embeddings-model"] = rag.DEFAULT_EMBEDDINGS_MODEL

def file_size_str(size, decimal_places=2):
    for unit in ['B', 'KiB', 'MiB', 'GiB', 'TiB', 'PiB']:
        if size < 1024.0 or unit == 'PiB':
            return f"{size:.{decimal_places}f} {unit}"
        size /= 1024.0
    return None

def split_stream(stream, ctx):
    for chunk in stream:
        if "answer" in chunk:
            yield chunk["answer"]
        if "context" in chunk:
            ctx.extend(chunk["context"])

def compat_messages(msgs):
    compat = []
    for msg in msgs:
        match msg.role:
            case rag.Message.USER_ROLE:
                compat.append(HumanMessage(content=msg.content))
            case rag.Message.ASSISTANT_ROLE:
                compat.append(AIMessage(content=msg.content))
    return compat
    
def main():
    model = load_prompt_model(DEFAULT_PROMPT_MODEL)
    db = load_rag_database()
    coll_id = load_selected_collection()
    vectorstore = load_selected_vectorstore()
    temp = load_temperature()
    seed = load_seed()
    llm = ChatOllama(model=model, keep_alive="-1", temperature=temp)
    retrieval_qa_chat_prompt = hub.pull("langchain-ai/retrieval-qa-chat")
    combine_docs_chain = create_stuff_documents_chain(
        llm, retrieval_qa_chat_prompt
    )

    log_session_state()

    st.set_page_config(
        page_title="Demo CENACE",
        page_icon="⚡️",
    )

    with st.sidebar:
        st.title("Colecciones")
        with st.container():
            for id, coll in db.index.items():
                st.button(
                    label=coll.name,
                    key=id,
                    help=coll.description,
                    type="secondary",
                    disabled=(coll_id == str(coll.id)),
                    use_container_width=True,
                    on_click=handle_select_collection,
                    args=(id,),
                )
        st.divider()
        with st.expander("Nueva colección"):
            with st.form("new-collection"):
                st.text_input("Nombre", key="new-coll-name")
                st.text_area("Descripción", key="new-coll-description")
                st.selectbox(
                    "Modelo de embeddings",
                    options=rag.EMBEDDINGS_MODELS,
                    key="new-coll-embeddings-model",
                )
                st.form_submit_button(
                    "⚡️",
                    type="primary",
                    on_click=handle_create_collection,
                )

    # Main content
    if coll_id is None:
        st.header("Help Desk")
        st.image("https://www.gob.mx/cms/uploads/identity/image/26465/background_cover-gob-abril-ok.jpg")
        st.markdown(
            """
            #### 1. Crea colecciones de documentos
            
            Haz clic en **Nueva colección** del menú de la
            izquierda. Luego especifica el *nombre* y *descripción* de
            la colección de documentos.
            
            #### 2. Agrega documentos PDF a una colección
            
            Selecciona una colección de la lista en el menú de la
            izquierda. Luego visita la pestaña de **Docs** para
            incorporar documentos a tu colección.
            
            #### 3. Chatea con un LLM sobre una colección
            
            En la pestaña de **Chat** podrás interactuar con un modelo
            de lenguaje para discutir sobre el contenido de tu
            colección de documentos.
            """
        )
        return
        
    
    assert db is not None
    assert coll_id in db.index
    coll = db.index[coll_id]
    st.header(coll.name)
    st.caption(coll.id)
    st.caption(coll.embeddings_model)
    st.markdown(f"*{coll.description}*")
    btn_cols = st.columns([0.25, 0.25, 0.5])
    if btn_cols[0].button("❎ Cerrar"):
        set_selected_collection(None)
        set_selected_vectorstore(None)
        st.rerun()
    if btn_cols[1].button("🔥 Eliminar"):
        if coll.embeddings_store_path is not None:
            shutil.rmtree(coll.embeddings_store_path, ignore_errors=True)
            shutil.rmtree(f"./{coll.id}_docs", ignore_errors=True)
        del db.index[str(coll.id)]
        set_selected_collection(None)
        set_selected_vectorstore(None)
        st.rerun()
    docs_tab = "🔍️ Docs"
    chat_tab = "🗣️ Chat"
    selected_tab = btn_cols[2].selectbox(
        "Vista",
        (docs_tab, chat_tab),
        label_visibility="collapsed",
    )
    st.divider()

    if selected_tab == docs_tab:
        if vectorstore is not None:
            col1, col2 = st.columns(2)
            col1.metric("Documentos", len(coll.files))
            col2.metric("Segmentos", vectorstore.index.ntotal)
        uploaded_files = st.file_uploader("⚙️ Cargar documentos", type=["pdf"], accept_multiple_files=True)
        paths = []
        add_doc_disabled = uploaded_files is None or len(uploaded_files)==0
        if st.button("Incorporar documentos", disabled=add_doc_disabled):
            with st.spinner("Incorporando documentos..."):
                st.warning("Incorporando documentos...")
                assert uploaded_files is not None
                for file in uploaded_files:
                    path = f"./{coll.id}_docs/{file.name}"
                    if os.path.exists(path):
                        st.error(f"¡El documento {file.name} ya existe!")
                        continue
                    with open(path, "wb") as fp:
                        fp.write(file.read())
                    paths.append(path)
            store = coll.add_files(paths)
            set_selected_vectorstore(store)
            db.save(DEFAULT_DATABASE_PATH)
                        
        with st.container(height=500):
            st.header("Documentos")
            if len(coll.files) == 0:
                st.write("No hay documentos en la colección")
            else:
                for file in coll.files:
                    st.divider()
                    st.markdown(
                        f"""
                        📄 **{file.path}**

                        *{file_size_str(file.size)}* / Última modificación: *{file.modtime:%Y-%m-%d}*
                        """
                    )
    elif selected_tab == chat_tab:
        if st.button("🧹 Borrar"):
            coll.history = []
            db.save(DEFAULT_DATABASE_PATH)
            st.rerun()
        chat_container = st.container(height=350, border=True)
        with chat_container:
            for msg in coll.history:
                match msg.role:
                    case rag.Message.USER_ROLE:
                        with st.chat_message("Human"):
                            st.markdown(msg.content)
                    case rag.Message.ASSISTANT_ROLE:
                        with st.chat_message("Ai"):
                            st.markdown(msg.content)
        chat_cols = st.columns([0.7, 0.3])
        if temp_val := chat_cols[0].slider("🌶️ Temperatura", 0.0, 2.0, temp, 0.01):
            set_temperature(temp_val)
        if seed_val := chat_cols[1].number_input("🌱 Semilla", -1, 999999, seed, 1):
            set_seed(seed_val)
        if user_input := st.chat_input("Escribe tu consulta", disabled=(vectorstore is None)):
            with chat_container.chat_message("Human"):
                st.markdown(user_input)

            ctx = []
            with chat_container.chat_message("Ai"):
                assert vectorstore is not None
                chain = create_retrieval_chain(vectorstore.as_retriever(), combine_docs_chain)
                stream = chain.stream(
                    {"input": user_input, "chat_history": compat_messages(coll.history)},
                )
                
                assistant_output = st.write_stream(
                    split_stream(stream, ctx)
                )
                # log(0, "* Context:", ctx)
                coll.add_message(rag.Message(rag.Message.USER_ROLE, user_input))
                coll.add_message(rag.Message(rag.Message.ASSISTANT_ROLE, assistant_output))
            db.save(DEFAULT_DATABASE_PATH)
            if len(ctx) > 0:
                with chat_container.expander("Referencias"):
                    md_refs = ""
                    for doc in ctx:
                        doc_src = doc.metadata["source"]
                        doc_page = doc.metadata["page"]
                        needle = f"{coll.id}_docs"
                        pdfname = doc_src[doc_src.find(needle) + len(needle) + 1:]
                        md_refs += f"""
                        - 📑 En {pdfname} página {doc_page}"""
                    st.markdown(md_refs)
                        
                

if __name__ == "__main__":
    main()
