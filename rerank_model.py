from transformers import AutoModelForSequenceClassification, AutoTokenizer
import os
import torch

from bm25_retriver import BM25
from pdf_parse import DataProcess
from config import *

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
        scores = scores.detach().cpu().clone().numpy()
        response = [doc for score, doc in sorted(zip(scores, docs), reverse=True, key=lambda x: x[0])]
        torch.cuda.empty_cache()
        return response
if __name__ == "__main__":
    bge_reranker_large = os.path.join(".", "pre_train_model", "bge-reranker-large")
    rerank = reRankLLM(bge_reranker_large)