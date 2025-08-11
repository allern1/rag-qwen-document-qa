from transformers import AutoModelForSequenceClassification, AutoTokenizer
import os
import torch

from bm25_retriever import BM25
from pdf_parse import DataProcess
from config import *

# 重排序模型（如 bge-reranker-large）的主要任务是：
# 计算相关性得分：对每一对（query, doc）计算一个相关性得分。
# 优化得分：通过复杂的特征提取和学习，使得得分能够更准确地反映文档与查询的相关性。

# 这种设置特别适用于单机训练场景，尤其是在资源有限的环境中，禁用并行处理可以显著提升系统的稳定性和性能
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# 释放gpu上没有用到的显存以及显存碎片
# def torch_gc():
#     if torch.cuda.is_available():
#         with torch.cuda.device()

# 加载rerank模型,即加载重排序模型
class reRankLLM(object):
    def __init__(self, model_path, max_length = 512):
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_path)
        self.model.eval()
        self.model.half()
        self.model.cuda()
        self.max_length = max_length

    # 输入文档对，返回每一对（query,doc)的相关得分，并从大到小排序
    def predict(self, query, docs):
        pairs = [(query, doc.page_content) for doc in docs]
        inputs = self.tokenizer(pairs, padding=True, truncation=True, return_tensors="pt", max_length=self.max_length).to("cuda")
        with torch.no_grad():
            # logits 是模型的原始输出，通常是一个张量，表示每个文档与查询的相关性得分。
            scores = self.model(**inputs).logits
        # detach()：节省内存，避免不必要的梯度计算，
        # 虽然 PyTorch 张量可以在 GPU 上进行高效的计算，但某些操作（如转换为 NumPy 数组）需要在 CPU 上进行。
        # NumPy 是 Python 中广泛使用的科学计算库，许多数据处理和排序操作更方便在 NumPy 数组上进行。例如，sorted() 函数需要一个 Python 原生的列表或数组作为输入，而不能直接处理 PyTorch 张量。
        scores = scores.detach().cpu().clone().numpy()
        # 模型输出未排序：模型的输出 logits 是一个张量，表示每个文档与查询的相关性得分，但这些得分的顺序与输入文档的顺序一致，并没有进行排序。
        response = [doc for score, doc in sorted(zip(scores, docs), reverse=True, key=lambda x: x[0])]
        torch.cuda.empty_cache()
        return response
if __name__ == "__main__":
    bge_reranker_large = os.path.join(".", "pre_train_model", "bge-reranker-large")
    rerank = reRankLLM(bge_reranker_large)