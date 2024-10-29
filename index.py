from llama_index.core import SimpleDirectoryReader, VectorStoreIndex, StorageContext, load_index_from_storage
import os

def load_index(index_dir, data_dir):
    """
    Load or create an index from documents in the specified directory.

    If the index directory does not exist, it reads documents from the data directory,
    creates a new index, and persists it. If the index directory exists, it loads the
    index from storage.

    """
    if not os.path.exists(index_dir):
        documents = SimpleDirectoryReader(input_dir=data_dir, recursive=True).load_data()
        index = VectorStoreIndex.from_documents(documents)
        index.storage_context.persist(persist_dir=index_dir)
    else:
        storage_context = StorageContext.from_defaults(persist_dir=index_dir)
        index = load_index_from_storage(storage_context)
    return index