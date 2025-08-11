#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
PDF 文本提取 + 多策略切块工具（PyMuPDF 版）
对外接口与旧版完全一致，可无痛替换。
依赖: PyMuPDF
"""
import fitz  # PyMuPDF
import re

# ----------------------------------------------------------------------
# 主类: DataProcess
# ----------------------------------------------------------------------
class DataProcess(object):
    def __init__(self, pdf_path: str):
        self.pdf_path = pdf_path
        self.data = []          # 最终结果
        self._seen = set()

    # 以下 SlidingWindow / Datafilter 与旧版完全一致，直接复制 ------
    def SlidingWindow(self, sentences, kernel: int = 512, overlap: int = 50):
        chunks, current = [], ""
        for sent in sentences:
            sent = sent.strip()
            if not sent:
                continue
            if not sent[-1] in "。！？.!?":
                sent += "。"
            if len(sent) > kernel:
                for i in range(0, len(sent), kernel):
                    chunks.append(sent[i:i + kernel])
                continue
            if len(current + sent) > kernel:
                if current:
                    chunks.append(current)
                current = current[-overlap:] if overlap else ""
            current += sent
        if current:
            chunks.append(current)
        for c in chunks:
            if c not in self._seen:
                self._seen.add(c)
                self.data.append(c)
        return chunks

    def Datafilter(self, line: str, header: str, pageid: int, max_seq: int = 1024):
        if len(line) < 2:
            return
        parts = re.split(r'[■•\t。！？.!?]', line)
        for p in parts:
            p = re.sub(r'[,\n\t]', '', p).strip()
            if 1 < len(p) < max_seq:
                item = f"{header}|{p}" if header else p
                if item not in self._seen:
                    self._seen.add(item)
                    self.data.append(item)

    # ------------------------------------------------------------------
    # PyMuPDF 版：提取当前页一级标题
    # 策略：取字号最大的 span 作为标题候选
    # ------------------------------------------------------------------
    def GetHeader(self, page: fitz.Page) -> str:
        try:
            blocks = page.get_text("dict")["blocks"]
        except Exception:
            return None

        header_text, max_size = None, 0
        for b in blocks:
            for line in b.get("lines", []):
                for span in line.get("spans", []):
                    txt, size = span["text"].strip(), span["size"]
                    if "目录" in txt or "....." in txt:
                        return None
                    if size > max_size:
                        max_size, header_text = size, txt
        return header_text

    # ------------------------------------------------------------------
    # PyMuPDF 版：按页提取「块」+ 标题组合
    # ------------------------------------------------------------------
    def ParseBlock(self, max_seq: int = 1024):
        doc = fitz.open(self.pdf_path)
        for pid, page in enumerate(doc, 1):
            header = self.GetHeader(page)
            if not header:
                continue

            blocks = page.get_text("dict")["blocks"]
            buf, last_size = "", 0
            for b in blocks:
                for line in b.get("lines", []):
                    for span in line.get("spans", []):
                        txt, size = span["text"], span["size"]
                        if txt in {"□", "•"}:
                            continue
                        if txt in {"警告！", "注意！", "说明！"}:
                            if buf:
                                self.Datafilter(buf, header, pid, max_seq)
                            buf = ""
                        elif abs(last_size - size) < 1e-5:
                            buf += txt
                        else:
                            if buf and len(buf) > 15:
                                self.Datafilter(buf, header, pid, max_seq)
                            buf, last_size = txt, size
            if buf:
                self.Datafilter(buf, header, pid, max_seq)
        doc.close()

    # ------------------------------------------------------------------
    # PyMuPDF 版：单页规则切块
    # ------------------------------------------------------------------
    def ParseOnePageWithRule(self, max_seq: int = 512, min_len: int = 6):
        doc = fitz.open(self.pdf_path)
        for pid, page in enumerate(doc, 1):
            text = page.get_text()
            lines = [ln.strip() for ln in text.splitlines()]
            page_text = "".join(ln for ln in lines
                                if ln and not ln.isdigit()
                                and "................" not in ln
                                and "目录" not in ln)
            if len(page_text) < min_len:
                continue
            if len(page_text) <= max_seq:
                self.Datafilter(page_text, "", pid, max_seq)
            else:
                sents = [s for s in re.split(r'[。！？.!?]', page_text) if s.strip()]
                self.SlidingWindow(sents, kernel=max_seq)
        doc.close()

    # ------------------------------------------------------------------
    # PyMuPDF 版：整文档滑窗切块
    # ------------------------------------------------------------------
    def ParseAllPage(self, max_seq: int = 512, min_len: int = 6):
        doc = fitz.open(self.pdf_path)
        all_text = ""
        for page in doc:
            txt = page.get_text() or ""
            lines = [ln.strip() for ln in txt.splitlines()]
            all_text += "".join(ln for ln in lines
                                if ln and not ln.isdigit()
                                and "................" not in ln
                                and "目录" not in ln)
        doc.close()

        if len(all_text) >= min_len:
            sents = [s for s in re.split(r'[。！？.!?]', all_text) if s.strip()]
            self.SlidingWindow(sents, kernel=max_seq)

# ----------------------------------------------------------------------
# 快速测试（与旧版完全一致）
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
    with open("all_text.txt", "w", encoding="utf-8") as f:
        for line in dp.data:
            f.write(line.strip() + "\n")