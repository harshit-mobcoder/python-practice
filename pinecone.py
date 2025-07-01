import os
import pinecone
from llama_index.core.indices import VectorStoreIndex
from llama_index.vector_stores.pinecone import PineconeVectorStore
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.core.schema import TextNode


def main():
    openai_api_key = os.environ.get("OPENAI_API_KEY")
    pinecone_api_key = os.environ.get("PINECONE_API_KEY")
    pinecone_env = os.environ.get("PINECONE_ENVIRONMENT", "us-east1-gcp")

    if not openai_api_key or not pinecone_api_key:
        raise ValueError("OPENAI_API_KEY and PINECONE_API_KEY must be set in environment")

    pinecone.init(api_key=pinecone_api_key, environment=pinecone_env)
    index_name = "gyaan-ai"

    if index_name not in pinecone.list_indexes():
        pinecone.create_index(index_name, dimension=1536)

    index = pinecone.Index(index_name)

    vector_store = PineconeVectorStore(pinecone_index=index)

    # create the index wrapper from the vector store
    vs_index = VectorStoreIndex.from_vector_store(
        vector_store, embed_model=OpenAIEmbedding()
    )

    # populate index with some sample data if it's empty
    if not index.describe_index_stats().get("total_vector_count"):
        nodes = [TextNode(text="Hello world"), TextNode(text="How are you?")]
        vs_index.insert_nodes(nodes)

    query_engine = vs_index.as_query_engine()
    response = query_engine.query("Hello?")
    print(response)


if __name__ == "__main__":
    main()
