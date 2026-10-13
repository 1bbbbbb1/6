import os
os.environ["DASHSCOPE_BASE_URL"] = "https://dashscope.aliyuncs.com/api/v1"

from abc import ABC, abstractmethod
from typing import Optional
from langchain_core.embeddings import Embeddings
from langchain_community.chat_models.tongyi import BaseChatModel
from langchain_community.embeddings import DashScopeEmbeddings
from langchain_community.chat_models.tongyi import ChatTongyi
from utils.config_handler import rag_conf

import dashscope
dashscope.api_key = "sk-4897fb37e99240f1a9a1f41ba0b08808"

class BaseModelFactory(ABC):
    @abstractmethod
    def generator(self) -> Optional[Embeddings | BaseChatModel]:
        pass


class ChatModelFactory(BaseModelFactory):
    def generator(self) -> Optional[Embeddings | BaseChatModel]:
        return ChatTongyi(
            model=rag_conf["chat_model_name"],
            dashscope_api_key="sk-4897fb37e99240f1a9a1f41ba0b08808"
        )


class EmbeddingsFactory(BaseModelFactory):
    def generator(self) -> Optional[Embeddings | BaseChatModel]:
        return DashScopeEmbeddings(
            model=rag_conf["embedding_model_name"],
            dashscope_api_key="sk-4897fb37e99240f1a9a1f41ba0b08808"
        )


chat_model = ChatModelFactory().generator()
embed_model = EmbeddingsFactory().generator()
