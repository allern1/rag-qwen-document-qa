'''
这段代码实现了一个基于多模态和多策略的问答系统。它的主要功能包括：
1. 数据预处理：从 PDF 文件中提取文本数据。
2. 向量召回：使用 Faiss 向量检索技术，根据文本嵌入向量召回相关文档。
3. BM25 召回：使用 BM25 算法召回相关文档。
4. 重排序（Re-ranking）：使用 reRankLLM 模型对召回的文档进行重排序。
5. 答案生成：结合多种召回结果，生成高质量的答案。
6. 评测：对测试问题生成答案，并保存结果。
'''
import json

from langchain_core.prompts import ChatPromptTemplate
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_retrieval_chain

import time
from rerank_model import reRankLLM
from faiss_retriever import FaissRetriever
from bm25_retriever import BM25
from pdf_parse import DataProcess
from hf_model import ChatLLM

# 获取Langchain的工具链
def get_qa_chain(llm, vector_store, prompt_template):
    """
    创建一个基于检索的问答链（现代 LangChain 写法）
    
    Args:
        llm: 大语言模型实例（如 ChatOpenAI, QwenChat, etc.）
        vector_store: 向量数据库（如 Chroma, FAISS）
        prompt_template: 包含 {context} 和 {input} 的模板字符串
    
    Returns:
        可调用的 retrieval chain
    """
    # 1. 创建检索器（设置 top_k=10）
    retriever = vector_store.as_retriever(search_kwargs={"k": 10})

    # 2. 构建提示模板（使用 ChatPromptTemplate）
    # 注意：新版本中通常使用 {input} 而不是 {question}
    prompt = ChatPromptTemplate.from_template(prompt_template)
    # 示例模板：
    # prompt_template = """根据以下上下文回答问题：
    # {context}
    # 问题: {input}"""

    # 3. 创建文档合并链（负责把检索到的文档 + 问题 给 LLM）
    combine_docs_chain = create_stuff_documents_chain(llm, prompt)

    # 4. 创建最终的检索链
    retrieval_chain = create_retrieval_chain(retriever, combine_docs_chain)

    return retrieval_chain

# 构造提示， 根据merged faiss和bm25的召回结果返回答案
def get_emb_bm25_merge(faiss_context, bm25_context, query):
    """
    将 FAISS（向量）和 BM25（关键词）检索结果合并，生成一个用于大模型问答的提示模板。
    Args:
        faiss_context: FAISS 检索返回的结果，格式为 [(Document, score), ...]
        bm25_context: BM25 检索返回的结果，格式为 [Document, ...]
        query: 用户提出的问题（字符串）
    Returns:
        str: 格式化后的 prompt 字符串，包含两类上下文信息和问题
    """
    # 定义拼接内容的最大长度（防止 prompt 过长导致 token 超限）
    max_length = 2500

    # BM25 检索通常只返回文档本身，而 FAISS 返回的是（文档, 相似度分数）元组。

    # 用于存储 FAISS 检索到的文档内容
    emb_ans = ""
    
    # 计数器，控制最多取前 6 个 FAISS 结果
    cnt = 0
    for doc, _ in faiss_context:
        cnt = cnt + 1
        if cnt > 6:
            break
        if len(emb_ans + doc.page_content) > max_length:
            break
        emb_ans = emb_ans + doc.page_content

    bm25_ans = ""
    cnt = 0
    for doc in bm25_context:
        cnt = cnt + 1
        if cnt > 6:
            break
        if len(bm25_ans + doc.page_content) > max_length:
            break
        emb_ans = bm25_ans + doc.page_content

    # emb_ans + bm25_ans 的意义是：融合“语义理解”和“关键词匹配”的优势，提供更全面、更鲁棒的上下文，提升问答准确率
    # 这叫做 混合检索（Hybrid Retrieval），是当前 RAG（检索增强生成）系统的最佳实践之一
    promt_template = """"基于以下已知信息，简洁和专业的来回答用户的问题。
                                如果无法从中得到答案，请说 "无答案"或"无答案"，不允许在答案中添加编造成分，答案请使用中文。
                                已知内容为吉利控股集团汽车销售有限公司的吉利用户手册:
                                1： {emb_ans}
                                2: {bm25_ans}
                                问题：
                                {question}""".format(emb_ans=emb_ans, bm25_ans=bm25_ans, question=query)
    return promt_template

def get_rerank(emb_ans, query):

    prompt_template = """基于以下已知信息，简洁和专业的来回答用户的问题。
                                如果无法从中得到答案，请说 "无答案"或"无答案" ，不允许在答案中添加编造成分，答案请使用中文。
                                已知内容为吉利控股集团汽车销售有限公司的吉利用户手册:
                                1: {emb_ans}
                                问题:
                                {question}""".format(emb_ans=emb_ans, question = query)
    return prompt_template

def question(text, llm, vector_store, prompt_template):
    chain = get_qa_chain(llm, vector_store, prompt_template)

    response = chain({"query": text})
    return response

def reRank(rerank, top_k, query, bm25_ams, faiss_ans):
    items = []
    max_length = 4000
    for doc, _ in faiss_ans:
        items.append(doc)
    items.extend(bm25_ams)
    rerank_ans = rerank.predict(query, items)
    rerank_ans = rerank_ans[: top_k]
    emb_ans = ""
    for doc in rerank_ans:
        if len(emb_ans + doc.page_content) > max_length:
            break
        emb_ans = emb_ans + doc.page_content
    return emb_ans

if __name__ == "__main__":
    start = time.time()
    base = "."
    qwen = base + "/pre_train_model/Qwen_0.6B"
    m3e =  base + "/pre_train_model/m3e-large"
    bge_reranker_large = base + "/pre_train_model/bge-reranker-large"

    # 解析pdf文档，构造数据
    dp =  DataProcess(pdf_path = base + "/data/train_a.pdf")
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
    print("data load ok")

    # Faiss 召回
    faissretriever = FaissRetriever(m3e, data)
    vector_store = faissretriever.vectors_store

    # BM25 召回
    bm25 = BM25(data)
    print("bm25 load ok")

    # llm大模型
    llm = ChatLLM(qwen)
    print("llm qwen load ok")

    # reRank模型
    rerank = reRankLLM(bge_reranker_large)
    print("rerank model load ok")

    # 对每一条测试问题，做答案生成处理
    with open(base + "/data/demo.json", "r") as f:
        jdata = json.load(f)
        print(len(jdata))
        max_length = 4000
        for idx, line in enumerate(jdata):
            query = line["question"]

            # faiss召回topk
            """"faiss_context 的结构可能类似于：
            [
            ("text1", 0.95),
            ("text2", 0.88),
            ("text3", 0.75),]
            """
            faiss_context = faissretriever.GetTopK(query, 15)
            # 提取与查询最相似的文本的相似度分数（faiss_min_score）
            faiss_min_score = 0.0
            if len(faiss_context) > 0:
                faiss_min_score = faiss_context[0][1]
            cnt = 0
            emb_ans = ""
            for doc, score in faiss_context:
                cnt = cnt + 1
                # 最长选择max length
                if len(emb_ans + doc.page_content) > max_length:
                    break
                # 最长选择max length
                if cnt > 6:
                    break
                emb_ans = emb_ans + doc.page_content
            
            # bm2.5召回topk
            bm25_context = bm25.GetBM25TopK(query, 15)
            bm25_ans = ""
            cnt = 0
            for doc in bm25_context:
                cnt = cnt + 1
                if len(bm25_ans + doc.page_content) > max_length:
                    break
                if cnt > 6:
                    break
                bm25_ans = bm25_ans + doc.page_content

            # 构造合并bm25召回和向量召回的prompt
            emb_bm25_merge_inputs = get_emb_bm25_merge(faiss_context, bm25_context, query)

            # 构造bm25召回的prompt
            bm25_inputs = get_rerank(bm25_ans, query)

            # 构造向量召回的prompt
            emb_inputs = get_rerank(emb_ans, query)

            # rerank召回的候选，并按照相关性得分排序
            rerank_ans = reRank(rerank, 6, query, bm25_context, faiss_context)

            # 构造得到rerank后生成答案的prompt
            rerank_inputs = get_rerank(rerank_ans, query)

            batch_input = []
            batch_input.append(emb_bm25_merge_inputs)
            batch_input.append(bm25_inputs)
            batch_input.append(emb_inputs)
            batch_input.append(rerank_inputs)

            # 执行batch推理
            batch_output = llm.infer(batch_input)
            line["answer_1"] = batch_output[0] # 合并两路召回的结果
            line["answer_2"] = batch_output[1] # bm召回的结果
            line["answer_3"] = batch_output[2] # 向量召回的结果
            line["answer_4"] = batch_output[3] # 多路召回重排序后的结果
            line["answer_5"] = emb_ans
            line["answer_6"] = bm25_ans
            line["answer_7"] = rerank_ans
            # 如果faiss检索跟query的距离高于500，输出无答案
            if(faiss_min_score >500):
                line["answer_5"] = "无答案"
            else:
                line["answer_5"] = str(faiss_min_score)

        # 保存结果，生成submission文件
        json.dump(jdata, open(base + "/data/result.json", "w", encoding='utf-8'), ensure_ascii=False, indent=2)
        end = time.time()
        print("cost time: " + str(int(end-start)/60))