# sipit/models/wrappers.py
import torch

class LMWrapper:
    """
    HFモデルをラップ。forwardで (hidden_states, logits) を返す。
    """
    def __init__(self, model, tokenizer):
        self.model = model
        self.tokenizer = tokenizer

    def encode(self, text: str):
        return self.tokenizer(text, return_tensors="pt")

    @torch.inference_mode()
    def forward(self, input_ids, attention_mask=None):
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True,
            return_dict=True
        )
        return outputs.hidden_states, outputs.logits

    @torch.inference_mode()
    def get_last_hidden(self, text: str, layer_idx: int):
        inputs = self.encode(text)
        hs, _ = self.forward(**inputs)
        return hs[layer_idx][0, -1, :].detach().cpu()
