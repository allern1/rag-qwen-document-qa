# 把一段文本（以及它的元数据）封装成 LangChain 能够识别和处理的“标准文档对象”，后面才能喂给向量库、检索器、LLM 等组件。
from langchain_core.documents import Document
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from pdf_parse import DataProcess
import torch
import os

class FaissRetriever(object):
    """
    将文本数据编码为向量后，存入 FAISS 索引，
    后续可通过向量检索返回最相似的文档。
    """

    # ----------------------------------------------------------------------
    # 构造函数：一次性完成「模型加载 → 文本向量化 → 索引构建」
    # ----------------------------------------------------------------------
    def __init__(self, model_path, data):
        # 1️. 加载用于中文句向量生成的 BGE 模型
        #    指定放在 GPU 上（"cuda"），加快批量编码速度
        self.embeddings = HuggingFaceEmbeddings(
            model_name=model_path,
            model_kwargs={"device": "cuda"}  # 使用 GPU
        )

        # 2️. 将原始文本转换成 LangChain 的 Document 对象列表
        docs = []
        for idx, line in enumerate(data):
            line = line.strip("\n").strip()     # 去掉两端换行和空格
            words = line.split("\t")            # 按 tab 切分（如适用）
            # 3️. 构造 Document：正文=words[0]，元数据=行号
            docs.append(
                Document(
                    page_content=words[0],
                    metadata={"idx": idx}      # 元数据方便后续定位原文
                )
            )

        # 4️. 调用 FAISS 接口：把文档列表一次性编码成向量并建索引
        self.vectors_store = FAISS.from_documents(docs, self.embeddings)

        # 5️. 释放显存
        #    删除 embeddings 对象 → 触发 Python 垃圾回收
        del self.embeddings
        # 6️. 手动清空 PyTorch 显存缓存，防止长期占用 GPU 内存
        torch.cuda.empty_cache()

    # 获取top-K分数最高的文档块
    def GetTopK(self, query, k: int=5):
        context = self.vectors_store.similarity_search(query, k=k)
        return context

    # 返回faiss向量检索对象
    def GetvectorStore(self):
        return self.vectors_store  
    

if __name__ == "__main__":
    base = "."
    model_name = model_name = os.path.join(base, "pre_train_model", "m3e-large")
    dp = DataProcess(pdf_path = os.path.join(base, "data", "train_a.pdf"))
    dp.ParseBlock(max_seq = 1024)
    dp.ParseBlock(max_seq = 512)
    print(len(dp.data))
    dp.ParseAllPage(max_seq = 256)
    dp.ParseAllPage(max_seq = 512)
    print(len(dp.data))
    dp.ParseOnePageWithRule(max_seq = 256)
    dp.ParseOnePageWithRule(max_seq = 512)
    print(len(dp.data))
    data = dp.data

    faissretriever = FaissRetriever(model_name, data)
    faiss_ans = faissretriever.GetTopK("如何预防新冠肺炎", 6)
    print(faiss_ans)
    faiss_ans = faissretriever.GetTopK("交通事故如何处理", 6)
    print(faiss_ans)
    faiss_ans = faissretriever.GetTopK("吉利集团的董事长是谁", 6)
    print(faiss_ans)
    faiss_ans = faissretriever.GetTopK("吉利汽车语音组手叫什么", 6)
    print(faiss_ans)