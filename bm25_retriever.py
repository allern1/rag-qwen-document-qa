from langchain_community.retrievers import BM25Retriever
from langchain_core.documents import Document
from pdf_parse import DataProcess
import jieba

class BM25(object):

    def __init__(self, documents):
        docs = []               # 用于索引的分词后文档
        full_docs = {}          # 方案 2：用 dict 保存 id -> 原始 Document，避免 idx 报错
        for idx, line in enumerate(documents):
            line = line.strip("\n").strip()
            if len(line) < 5:
                continue
            tokens = " ".join(jieba.cut_for_search(line))
            docs.append(Document(page_content=tokens, metadata={"id": idx}))

            # 原始文本按 "\t" 取第一段作为正文
            words = line.split("\t")
            full_docs[idx] = Document(page_content=words[0], metadata={"id": idx})

        self.documents = docs
        self.full_docs_map = full_docs   # 这里换成字典
        self.retriver = self.__init__bm25()

    def __init__bm25(self):
        return BM25Retriever.from_documents(self.documents)

    def GetBM25TopK(self, query, topk):
        self.retriver.k = topk
        query = " ".join(jieba.cut_for_search(query))
        docs = self.retriver.invoke(query)

        # 方案 2：直接通过 id 从字典里拿原始 Document，安全又高效
        result_docs = []
        for d in docs:
            doc_id = d.metadata["id"]
            # 如果初始化时该行被过滤掉，字典里就没有，默认跳过即可
            if doc_id in self.full_docs_map:
                result_docs.append(self.full_docs_map[doc_id])
        return result_docs
    
if __name__ == "__main__":

    # bm2.5
    dp =  DataProcess(pdf_path = "./data/train_a.pdf")
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
    bm25 = BM25(data)
    res = bm25.GetBM25TopK("座椅加热", 6)
    print(res)