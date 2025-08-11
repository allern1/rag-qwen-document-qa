# hf_model.py
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

class ChatLLM:
    def __init__(self, model_path: str):
        # 1. 加载 tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_path,
            trust_remote_code=True,
            padding_side="left"
        )
        # 如果 pad_token 不存在，就用 eos_token 代替
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        # 2. 加载模型（自动量化 / 自动设备映射）
        self.model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=torch.bfloat16,
            device_map="auto",
            trust_remote_code=True
        )

    def infer(self, questions):
        """
        批量推理
        :param questions: List[str]，用户问题列表
        :return: List[str]，模型回答列表
        """
        prompts = []
        for q in questions:
            # 构造对话 prompt（兼容 Qwen3）
            messages = [
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user",   "content": q}
            ]
            prompt = self.tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True
            )
            prompts.append(prompt)

        # 3. 批处理 tokenize
        inputs = self.tokenizer(
            prompts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=4096
        ).to(self.model.device)

        # 4. 生成
        with torch.no_grad():
            gen_ids = self.model.generate(
                **inputs,
                max_new_tokens=512,
                do_sample=True,  # 更自然
                temperature=0.7,
                top_p=0.8,
                eos_token_id=self.tokenizer.eos_token_id,
                pad_token_id=self.tokenizer.pad_token_id
            )

        # 5. 解码并去掉输入部分、特殊 token
        input_len = inputs["input_ids"].shape[1]   # prompt 实际长度
        new_ids = gen_ids[:, input_len:]           # 去掉 prompt
        answers = self.tokenizer.batch_decode(
            new_ids,
            skip_special_tokens=True
        )
        results = [ans.replace("<|im_end|>", "").replace("<|endoftext|>", "").strip()
                for ans in answers]

        return results