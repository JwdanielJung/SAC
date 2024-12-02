from transformers import AutoTokenizer, AutoModelForCausalLM
import torch


class Llamaguard:
    def __init__(self):
        self._model_id = "meta-llama/LlamaGuard-7b"
        # self._model_id = "meta-llama/Meta-Llama-Guard-2-8B"
        self._device = "cuda"
        self._tokenizer = AutoTokenizer.from_pretrained(self._model_id)
        self._model = AutoModelForCausalLM.from_pretrained(
            self._model_id, torch_dtype=torch.bfloat16, device_map=self._device
        )

    def _moderate(self, chat):
        input_ids = self._tokenizer.apply_chat_template(chat, return_tensors="pt").to(
            self._device
        )
        output = self._model.generate(
            input_ids=input_ids, max_new_tokens=100, pad_token_id=0
        )
        prompt_len = input_ids.shape[-1]
        return self._tokenizer.decode(output[0][prompt_len:], skip_special_tokens=True)

    def unsafe_classification(self, texts):

        results = [
            (
                0
                if self._moderate(
                    [
                        {"role": "user", "content": text},
                    ]
                )
                == "safe"
                else 1
            )
            for text in texts
        ]
        return results
