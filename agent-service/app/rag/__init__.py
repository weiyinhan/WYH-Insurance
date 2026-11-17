from langchain_ollama import OllamaEmbeddings
from langchain_milvus import Milvus, BM25BuiltInFunction
from app.core.config import settings
from app.rag.pipeline import RAGPipeline
from app.rag.retriever import ParentChunkRetriever

# 1.初始化向量模型
embeddings = OllamaEmbeddings(
    model="nomic-embed-text:v1.5",  # 使用本地已有的向量模型
    dimensions=768  # nomic-embed-text 的输出维度
)

# 2、初始化 VectorStore，会自动创建Collection
vector_store = Milvus(
    embedding_function=embeddings,
    collection_name="insurance_collection",
    builtin_function=BM25BuiltInFunction(analyzer_params={"type": "chinese"}),
    vector_field=["dense", "sparse"],
    connection_args={"uri": settings.rag.milvus_url},
    auto_id=True,
)

# 3.创建Pipeline
rag_pipeline = RAGPipeline(vector_store)

# 4.创建Retriever
retriever = ParentChunkRetriever(vector_store)


# 5.关闭资源
def close_rag():
    rag_pipeline.close()
    vector_store.client.close()

__all__ = ['pipeline', 'retriever', 'close_rag']