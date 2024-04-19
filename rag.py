from __future__ import annotations
from typing import Any, List, Iterable, Optional, Literal

import os
import json
from uuid import UUID, uuid4
from datetime import datetime, timezone

from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores.faiss import FAISS
from langchain_community.embeddings import OllamaEmbeddings

T_Path = str | os.PathLike

EMBEDDINGS_MODELS = [
    "nomic-embed-text:latest",
    "mxbai-embed-large:latest",
    "all-minilm:latest",
]

DEFAULT_EMBEDDINGS_MODEL = "nomic-embed-text:latest"

T_EmbeddingsModel = Literal[
    "nomic-embed-text:latest",
    "mxbai-embed-large:latest",
    "all-minilm:latest",
]

def file_modtime(path: T_Path):
    modtime_ts = os.path.getmtime(path)
    modtime_dt = datetime.fromtimestamp(modtime_ts)
    modtime_dtz = modtime_dt.astimezone()
    return modtime_dtz.astimezone(timezone.utc)


def file_size(path: T_Path):
    return os.stat(path).st_size

class Message:
    USER_ROLE = "user"
    ASSISTANT_ROLE = "assistant"
    def __init__(self, role, content):
        self.role = role
        self.content = content

    def to_dict(self):
        return { 
            "role": self.role,
            "content": self.content,
        }

    @staticmethod
    def from_dict(dict: Any) -> Message:
        assert "role" in dict
        assert "content" in dict
        assert dict["role"] in [Message.USER_ROLE, Message.ASSISTANT_ROLE]
        return Message(
            role = dict["role"],
            content = dict["content"],
        )

class File:
    NOT_EXISTS = "rag-file-not-exists"
    MODTIME_CHANGED = "rag-file-modtime-changed"
    SIZE_CHANGED = "rag-file-size-changed"
    OK = "rag-file-ok"

    def __init__(self, path: str, modtime: datetime, size: int):
        self.path = os.path.abspath(path)
        self.modtime = modtime
        self.size = size

    def check_integrity(self):
        path = self.path
        if not os.path.exists(path):
            return self.NOT_EXISTS
        if file_modtime(path) != self.modtime:
            return self.MODTIME_CHANGED
        if file_size(path) != self.size:
            return self.SIZE_CHANGED
        return self.OK

    def to_dict(self):
        return {
            "path": self.path,
            "modtime": self.modtime.isoformat(),
            "size": self.size,
        }

    @staticmethod
    def from_dict(dict: Any) -> File:
        assert "path" in dict
        assert "modtime" in dict
        assert "size" in dict
        return File(
            dict["path"], datetime.fromisoformat(dict["modtime"]), dict["size"]
        )

    @staticmethod
    def from_path(path: T_Path):
        path = os.path.abspath(path)
        assert os.path.exists(path)
        modtime = file_modtime(path)
        size = file_size(path)
        return File(path, modtime, size)

    def get_chunks(self):
        path = self.path
        loader = PyPDFLoader(path)
        chunks = loader.load_and_split()
        return chunks


class Collection:
    STORE_NOT_EXISTS = "rag-collection-store-not-exists"
    MISSING_EMBEDDINGS = "rag-collection-missing-embeddings"
    OK = "rag-collection-ok"

    def __init__(
        self,
        id: UUID,
        name: str,
        description: str,
        files: List[File] = [],
        embeddings_model: T_EmbeddingsModel = DEFAULT_EMBEDDINGS_MODEL,
        embeddings_store_path: Optional[str] = None,
        history: List[Message] = [],
    ):
        self.id = id
        self.name = name.strip()
        assert self.name != ""
        self.description = description.strip()
        assert self.description != ""
        self.files = files
        self.embeddings_model = embeddings_model
        self.embeddings = OllamaEmbeddings(model=self.embeddings_model)
        if embeddings_store_path is None:
            self.embeddings_store_path = None
        else:
            self.embeddings_store_path = os.path.abspath(embeddings_store_path)
        self.history = history

    def check_integrity(self):
        for file in self.files:
            status = file.check_integrity()
            if status != File.OK:
                return status, file
        path = self.embeddings_store_path
        if path is None:
            return self.MISSING_EMBEDDINGS, None
        if not os.path.exists(path):
            return self.STORE_NOT_EXISTS, path
        return self.OK

    def to_dict(self):
        return {
            "id": str(self.id),
            "name": self.name,
            "description": self.description,
            "files": [file.to_dict() for file in self.files],
            "embeddings_model": self.embeddings_model,
            "embeddings_store_path": self.embeddings_store_path,
            "history": [msg.to_dict() for msg in self.history],
        }

    @staticmethod
    def from_dict(dict: Any) -> Collection:
        assert "id" in dict
        assert "name" in dict
        assert "description" in dict
        assert "files" in dict
        assert "embeddings_model" in dict
        assert "embeddings_store_path" in dict
        assert "history" in dict
        return Collection(
            id=UUID(dict["id"]),
            name=dict["name"],
            description=dict["description"],
            files=[File.from_dict(fdict) for fdict in dict["files"]],
            embeddings_model=dict["embeddings_model"],
            embeddings_store_path=dict["embeddings_store_path"],
            history=[Message.from_dict(mdict) for mdict in dict["history"]],
        )

    @staticmethod
    def empty(
        name: str,
        description: str,
        embeddings_model: T_EmbeddingsModel = DEFAULT_EMBEDDINGS_MODEL,
    ):
        name = name.strip()
        assert name != ""
        description = description.strip()
        assert description != ""
        id = uuid4()
        os.mkdir(os.path.abspath(f"./{id}_docs"))
        return Collection(
            id=id,
            name=name,
            description=description,
            files=[],
            embeddings_model=embeddings_model,
            embeddings_store_path=None,
            history=[],
        )

    def generate_embeddings(self, files: List[File]) -> FAISS:
        files = list(files)
        assert len(files) > 0
        chunks = []
        for file in files:
            chunks.extend(file.get_chunks())
        store = FAISS.from_documents(chunks, self.embeddings)
        return store

    def load_store(self):
        path = self.embeddings_store_path
        if path is None:
            return None
        assert os.path.exists(path)
        store = FAISS.load_local(path, self.embeddings, allow_dangerous_deserialization=True)
        return store
        

    def add_files(self, paths: List[T_Path]):
        curr_paths = [file.path for file in self.files]
        new_files = []
        for path in paths:
            assert os.path.exists(path)
            if path in curr_paths:
                continue
            new_files.append(File.from_path(path))
        if len(new_files) == 0:
            return self.load_store()
        new_store = self.generate_embeddings(new_files)
        store = self.load_store()
        if store is None:
            store = new_store
            self.embeddings_store_path = os.path.abspath(f"{self.id}_store")
        else:
            store.merge_from(new_store)
        assert self.embeddings_store_path is not None
        store.save_local(self.embeddings_store_path)
        self.files.extend(new_files)
        return store

    def add_message(self, msg: Message):
        self.history.append(msg)
        
        
        
class Database:
    def __init__(self, index: dict[str, Collection]):
        self.index = index

    def save(self, path: T_Path):
        path = os.path.abspath(path)
        data = {}
        for id, collection in self.index.items():
            data[id] = collection.to_dict()
        with open(path, "w") as fp:
            json.dump(data, fp, indent="    ")

    @staticmethod
    def load(path: T_Path):
        path = os.path.abspath(path)
        assert os.path.exists(path)
        index = {}
        with open(path, "r") as fp:
            data = json.load(fp)
            for id, entry in data.items():
                index[id] = Collection.from_dict(entry)
        return Database(index)
