"""
memory_manager.py - 基于RAG的长期记忆管理模块
使用向量数据库存储和检索用户信息
"""

import os
import logging
from typing import List, Optional
import config

from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma

logger = logging.getLogger(__name__)


class MemoryManager:
    def __init__(self,
                 persist_dir: str = config.MEMORY_PERSIST_DIR,
                 collection_name: str = config.MEMORY_COLLECTION_NAME,
                 embedding_model: str = config.EMBEDDING_MODEL,
                 retrieval_k: int = config.MEMORY_RETRIEVAL_K,
                 device: str = config.MEMORY_DEVICE):
        """
        初始化记忆管理器

        :param persist_dir: 向量数据库持久化目录
        :param collection_name: 集合名称
        :param embedding_model: 嵌入模型名称（支持 HuggingFace 模型）
        :param retrieval_k: 每次检索返回的最相关记忆条数
        :param device: 运行设备 ('cpu' 或 'cuda')
        """
        self.persist_dir = persist_dir
        self.collection_name = collection_name
        self.retrieval_k = retrieval_k
        self.device = device

        # 创建持久化目录
        os.makedirs(persist_dir, exist_ok=True)

        # 加载嵌入模型
        logger.info(f"加载嵌入模型: {embedding_model}")
        self.embeddings = HuggingFaceEmbeddings(
            model_name=embedding_model,
            model_kwargs={'device': device},
            encode_kwargs={'normalize_embeddings': True}
        )

        # 初始化或加载向量数据库
        self.vectorstore = Chroma(
            collection_name=collection_name,
            embedding_function=self.embeddings,
            persist_directory=persist_dir
        )
        self.retriever = self.vectorstore.as_retriever(
            search_kwargs={"k": retrieval_k}
        )
        logger.info(f"记忆管理器初始化完成，数据库位置: {persist_dir}")

    def add_memory(self, text: str, metadata: Optional[dict] = None) -> None:
        """
        添加一条记忆到数据库

        :param text: 记忆文本内容
        :param metadata: 附加元数据（如时间戳、来源等）
        """
        if metadata is None:
            metadata = {"source": "user"}
        self.vectorstore.add_texts([text], metadatas=[metadata])
        self.vectorstore.persist()  # 立即持久化
        logger.debug(f"记忆已添加: {text}")

    def add_memories(self, texts: List[str], metadatas: Optional[List[dict]] = None) -> None:
        """
        批量添加记忆

        :param texts: 记忆文本列表
        :param metadatas: 元数据列表（长度需与 texts 一致）
        """
        if metadatas is None:
            metadatas = [{"source": "user"} for _ in texts]
        self.vectorstore.add_texts(texts, metadatas=metadatas)
        self.vectorstore.persist()
        logger.debug(f"批量添加 {len(texts)} 条记忆")

    def add_memory_unique(self, text: str, threshold: float = 0.95, metadata: Optional[dict] = None) -> bool:
        """
        添加记忆，如果数据库中已存在相似度超过阈值的文本，则不添加。
        返回 True 表示已添加，False 表示已存在未添加。
        """
        # 检索相似记忆
        docs_with_scores = self.vectorstore.similarity_search_with_relevance_scores(
            text, k=1
        )
        if docs_with_scores:
            doc, score = docs_with_scores[0]
            if score >= threshold:
                logger.debug(f"记忆已存在（相似度 {score:.3f}），跳过: {text}")
                return False

        # 不存在则添加
        if metadata is None:
            metadata = {"source": "user"}
        self.vectorstore.add_texts([text], metadatas=[metadata])
        self.vectorstore.persist()
        logger.info(f"新记忆已添加: {text}")
        return True

    def retrieve_memories(self, query: str) -> List[str]:
        """
        检索与查询相关的记忆

        :param query: 查询文本（通常是用户输入）
        :return: 相关记忆文本列表
        """
        docs = self.retriever.invoke(query)
        return [doc.page_content for doc in docs]

    def retrieve_with_scores(self, query: str) -> List[tuple]:
        """
        检索记忆并返回带相似度分数的结果

        :param query: 查询文本
        :return: 列表，每个元素为 (文本, 分数)
        """
        docs_with_scores = self.vectorstore.similarity_search_with_relevance_scores(query, k=self.retrieval_k)
        return [(doc.page_content, score) for doc, score in docs_with_scores]

    def delete_memory(self, ids: List[str]) -> None:
        """
        根据文档ID删除记忆
        注：Chroma 需要先获取文档ID才能删除，此方法需要配合 collection.get()
        为简化，可跳过或提供更复杂的实现
        """
        # 简单起见，暂不实现，或可调用 self.vectorstore.delete(ids)
        pass

    def clear_all(self) -> None:
        """清空所有记忆（谨慎使用）"""
        # 删除集合并重建
        self.vectorstore.delete_collection()
        self.vectorstore = Chroma(
            collection_name=self.collection_name,
            embedding_function=self.embeddings,
            persist_directory=self.persist_dir
        )
        self.retriever = self.vectorstore.as_retriever(
            search_kwargs={"k": self.retrieval_k}
        )
        self.vectorstore.persist()
        logger.warning("所有记忆已清空")