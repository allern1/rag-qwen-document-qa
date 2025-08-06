from langchain_community.retrievers import BM25Retriever
from langchain_core.documents import Document
from pdf_parse import DataProcess
# jieba 是为“基于词的模型”服务的
import jieba

class BM25(object):

    # 遍历文档，首先做分词，然后把分词后的文档和全文文档建立索引和映射关系
    def __init__(self, documents):
        # docs (用于索引),内容: 存储的是经过 jieba.cut_for_search(line) 分词并用空格连接后的字符串 tokens。目的: 专门用于构建 BM25 检索器的倒排索引
        # full_docs (用于返回):内容: 存储的是原始的、未分词的文本内容（即 words[0]，通常是每行文本中代表主要内容的部分）。目的: 用于在检索到相关文档后，返回给用户原始的、可读的完整内容
        # 两者通过相同的 metadata={"id": idx} 进行关联
        docs = []
        full_docs = []
        for idx, line in documents:
            # 清理文本行，去除多余的空白字符，得到干净的文本内容。
            line = line.strip("\n").strip()
            # 过滤掉过短的文本行。过短的行可能只是标题、分隔符、页码或无意义的片段，对检索没有帮助，甚至可能引入噪声。这是一个简单的文本质量过滤。
            if(len(line)<5):
                continue
            # 将清理后且长度不小于5的中文文本 line，使用 jieba 的“搜索引擎模式”进行分词，然后将分出的所有词语用空格连接起来，形成一个单一的字符串 tokens
            # jieba.cut_for_search(line)它在精确模式的基础上，对长词再次进行切分，以提高召回率。
            tokens = " ".join(jieba.cut_for_search(line))
            docs.append(Document(page_content=tokens, metadata={"id": idx}))
            words = line.split("\t")
            full_docs.append(Document(page_content=words[0], metadata={"id": idx}))
        self.documents = docs
        self.full_documents = full_docs
        self.retriver = self.__init__bm25()

    # 初始化BM25的知识库
    # 将 BM25Retriever 的初始化放在一个私有方法 __init__bm25 中，是一种良好的封装习惯。它清晰地表明了这个方法的唯一目的就是创建并返回检索器实例
    def __init__bm25(self):
        return BM25Retriever.from_documents(self.documents)
    
    # 获得得分在topk的文档和分数
    def GetBM25TopK(self, query, topk):
        # BM25Retriever 有一个属性 k，它决定了每次检索时返回多少个最相关的文档。
        self.retriver.k = topk
        # 确保了查询和文档的处理方式完全一致。
        query = " ".join(jieba.cut_for_search(query))
        ans_docs = self.retriver.get_relevant_documents(query)
        ans = []
        for line in ans_docs:
            ans.append(self.full_documents[line.metadata["id"]])
        return ans
    
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