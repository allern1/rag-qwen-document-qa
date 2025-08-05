#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
PDF 文本提取 + 多策略切块工具
功能:
  1. 按页提取标题+正文
  2. 按规则或滑窗方式将长文本切成 <= kernel 的文本块
依赖: pdfplumber, PyPDF2, re
"""
import pdfplumber               # 高精度文本/表格提取
from PyPDF2 import PdfReader    # 快速整页提取
import re                       # 正则切分/清洗

# ----------------------------------------------------------------------
# 主类: DataProcess
# ----------------------------------------------------------------------
class DataProcess(object):
    """
    参数:
        pdf_path: PDF 文件路径
    属性:
        data: 最终文本块列表（list，保持顺序）
        _seen: 内部 set，用于快速去重
    """

    def __init__(self, pdf_path: str):
        self.pdf_path = pdf_path
        self.data = []          # 最终结果
        self._seen = set()      # 用于去重，避免重复插入

    # ------------------------------------------------------------------
    # 1. 滑动窗口切块
    #    sentences: 句子列表（可以是长句或短句）
    #    kernel:    每个文本块最大字符数
    #    overlap:   字符级重叠长度
    # ------------------------------------------------------------------
    def SlidingWindow(self, sentences, kernel: int = 512, overlap: int = 50):
        """
        把句子列表切成 <= kernel 的文本块，块与块之间保留 overlap 字符。
        返回 chunks 列表，同时去重后追加到 self.data。
        """
        chunks = []             # 本次调用产生的新块
        current = ""            # 当前正在累积的块

        for sent in sentences:
            sent = sent.strip()
            if not sent:        # 跳过空句
                continue
            # 缺句号自动补上，避免拼接后无终止符
            if not sent[-1] in "。！？.!?":
                sent += "。"

            # 单句超长 → 直接按 kernel 强制截断
            if len(sent) > kernel:
                # 步长 stride 未用，此处按 kernel 截断
                for i in range(0, len(sent), kernel):
                    seg = sent[i:i + kernel]
                    chunks.append(seg)
                continue

            # 正常累积：若加上这句会超限，则先保存 current
            if len(current + sent) > kernel:
                if current:
                    chunks.append(current)
                # 保留 overlap 字符作为新块开头
                current = current[-overlap:] if overlap else ""

            current += sent

        # 剩余内容
        if current:
            chunks.append(current)

        # 去重后加入全局列表
        for c in chunks:
            if c not in self._seen:
                self._seen.add(c)
                self.data.append(c)
        return chunks

    # ------------------------------------------------------------------
    # 2. 单行文本过滤 & 切块（供 ParseBlock/OnePageWithRule 调用）
    #    line: 原始字符串
    #    header: 当前页一级标题（可留空）
    #    pageid: 页码（调试用）
    #    max_seq: 单块最大字符数
    # ------------------------------------------------------------------
    def Datafilter(self, line: str, header: str, pageid: int, max_seq: int = 1024):
        """
        把一行长文本按中英文标点/特殊符号切分，
        保留长度 1~max_seq 的子句，并去重后追加到 self.data
        """
        if len(line) < 2:
            return
        # 正则：支持中英文标点、项目符号、制表符
        parts = re.split(r'[■•\t。！？.!?]', line)
        for p in parts:
            p = re.sub(r'[,\n\t]', '', p).strip()
            if 1 < len(p) < max_seq:
                # 可选：带上页头信息
                item = f"{header}|{p}" if header else p
                if item not in self._seen:
                    self._seen.add(item)
                    self.data.append(item)

    # ------------------------------------------------------------------
    # 3. 提取当前页的「一级标题」
    # ------------------------------------------------------------------
    def GetHeader(self, page):
        """
        返回该页最上方的标题文本，用于后续拼接。
        无法识别时返回 None。
        """
        # pdfplumber 把一页 PDF 中的所有文字拆成“单个单词/字符块”，并返回一个包含每个文字块详细信息（文本、坐标、字体大小等）的列表。
        try:
            words = page.extract_words()
        except Exception:
            return None
        for w in words:
            txt = w["text"]
            # 目录/点线忽略
            if "目录" in txt or "....." in txt:
                return None
            # PDF 坐标系：top 越小越靠上
            if 17 < w["top"] < 20:
                return txt
        return words[0]["text"] if words else None

    # ------------------------------------------------------------------
    # 4. 按页提取「块」+ 标题组合
    # ------------------------------------------------------------------
    def ParseBlock(self, max_seq: int = 1024):
        """
        逐页读取，按字体大小差异切分块，再交给 Datafilter 处理。
        """
        with pdfplumber.open(self.pdf_path) as pdf:
            for pid, page in enumerate(pdf.pages):
                header = self.GetHeader(page)
                if not header:
                    continue

                words = page.extract_words(use_text_flow=True, extra_attrs=["size"])
                buf, last_size = "", 0
                for w in words:
                    txt, size = w["text"], w["size"]
                    # 跳过项目符号
                    if txt in {"□", "•"}:
                        continue
                    # 遇到警告/注意 → 先保存当前块
                    if txt in {"警告！", "注意！", "说明！"}:
                        if buf:
                            self.Datafilter(buf, header, pid, max_seq)
                        buf = ""
                    # 字号相同 → 同一段
                    elif abs(last_size - size) < 1e-5:
                        buf += txt
                    # 字号不同 → 新段
                    else:
                        if buf and len(buf) > 15:
                            self.Datafilter(buf, header, pid, max_seq)
                        buf, last_size = txt, size
                # 最后一段
                if buf:
                    self.Datafilter(buf, header, pid, max_seq)

    # ------------------------------------------------------------------
    # 5. 单页规则切块（先句号，再滑窗）
    # ------------------------------------------------------------------
    def ParseOnePageWithRule(self, max_seq: int = 512, min_len: int = 6):
        """
        逐页提取纯文本，若整页 < min_len 则丢弃；
        否则先按句号切句，再滑窗切块。
        """
        reader = PdfReader(self.pdf_path)
        for pid, page in enumerate(reader.pages):
            text = page.extract_text() or ""
            # 清理行
            lines = [ln.strip() for ln in text.splitlines()]
            page_text = "".join(ln for ln in lines
                                if ln and not ln.isdigit()
                                and "................" not in ln
                                and "目录" not in ln)
            if len(page_text) < min_len:
                continue
            # 短页直接保留
            if len(page_text) <= max_seq:
                self.Datafilter(page_text, "", pid, max_seq)
            else:
                # 先按句号切句
                sents = [s for s in re.split(r'[。！？.!?]', page_text) if s.strip()]
                self.SlidingWindow(sents, kernel=max_seq)

    # ------------------------------------------------------------------
    # 6. 整文档滑窗切块（所有页文本合并后切块）
    # ------------------------------------------------------------------
    def ParseAllPage(self, max_seq: int = 512, min_len: int = 6):
        """
        把整本 PDF 当成一个长文本，先清洗再滑窗切块。
        """
        reader = PdfReader(self.pdf_path)
        all_text = ""
        for page in reader.pages:
            txt = page.extract_text() or ""
            lines = [ln.strip() for ln in txt.splitlines()]
            all_text += "".join(ln for ln in lines
                                if ln and not ln.isdigit()
                                and "................" not in ln
                                and "目录" not in ln)
        if len(all_text) >= min_len:
            sents = [s for s in re.split(r'[。！？.!?]', all_text) if s.strip()]
            self.SlidingWindow(sents, kernel=max_seq)

# ----------------------------------------------------------------------
# 7. 快速测试
# ----------------------------------------------------------------------
if __name__ == "__main__":
    dp = DataProcess("./data/train_a.pdf")
    dp.ParseBlock(1024)
    dp.ParseBlock(512)
    dp.ParseAllPage(256)
    dp.ParseAllPage(512)
    dp.ParseOnePageWithRule(256)
    dp.ParseOnePageWithRule(512)

    print("最终文本块总数：", len(dp.data))
    # 写文件
    with open("all_text.txt", "w", encoding="utf-8") as f:
        for line in dp.data:
            f.write(line.strip() + "\n")