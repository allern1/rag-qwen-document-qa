from vllm import LLM, SamplingParams
from transformers import AutoTokenizer, AutoConfig
import time, os
os.environ["TOKENIZERS_PARALLELISM"] = "false"

def get_stop_ids(chat_format, tokenizer):
    if chat_format == "chatml":
        return [tokenizer.im_end_id, tokenizer.im_start_id, tokenizer.eos_token_id]
    return [tokenizer.eos_token_id]

class ChatLLM:
    def __init__(self, model_path: str):
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_path, trust_remote_code=True, padding_side="left"
        )
        config = AutoConfig.from_pretrained(model_path, trust_remote_code=True)
        self.chat_format = getattr(config, "chat_format", "chatml")
        self.max_window_size = getattr(config, "max_window_size", 6144)

        stop_ids = get_stop_ids(self.chat_format, self.tokenizer)
        self.sampling_params = SamplingParams(
            stop_token_ids=stop_ids,
            temperature=0,
            max_tokens=2000,
            repetition_penalty=1.05,
            use_beam_search=True,
            best_of=2,
            top_p=1.0,
            top_k=-1,
        )
        self.model = LLM(
            model=model_path,
            tokenizer=model_path,
            tensor_parallel_size=1,
            trust_remote_code=True,
            gpu_memory_utilization=0.6,
            dtype="bfloat16",
            max_model_len=4096,
            max_num_seqs=8,
        )

    def infer(self, questions):
        from qwen_generation_utils import make_context
        prompts = []
        for q in questions:
            prompt, _ = make_context(
                self.tokenizer,
                q,
                history=None,
                system="You are a helpful assistant.",
                max_window_size=self.max_window_size,
                chat_format=self.chat_format,
            )
            prompts.append(prompt)
        outputs = self.model.generate(prompts, self.sampling_params)
        results = [o.outputs[0].text.rstrip("<|im_end|>").rstrip("<|endoftext|>") for o in outputs]
        return results

if __name__ == "__main__":
    model_dir = "/root/autodl-tmp/codes/pre_train_model/Qwen-7B-Chat"
    t0 = time.time()
    llm = ChatLLM(model_dir)
    print("Model loaded in %.1fs" % (time.time() - t0))

    questions = ["吉利汽车座椅按摩", "吉利汽车语音助手唤醒", "自动驾驶功能介绍"]
    answers = llm.infer(questions)
    for q, a in zip(questions, answers):
        print(f"Q: {q}\nA: {a}\n{'-'*60}")
    print("Total cost %.1fs" % (time.time() - t0))