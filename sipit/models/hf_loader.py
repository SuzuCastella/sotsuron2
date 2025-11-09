# sipit/models/hf_loader.py
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

def load_model_and_tokenizer(model_name: str, device="cuda", dtype="float32", load_in_8bit=False):
    """
    任意のHF CausalLMをロード。hidden取得のためにconfig側を有効化。
    """
    print(f"🔹 Loading model: {model_name} on {device} ({dtype})")

    torch_dtype = {"float32": torch.float32, "float16": torch.float16, "bfloat16": torch.bfloat16}[dtype]

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch_dtype,
        device_map="auto" if device.startswith("cuda") else None,
        load_in_8bit=load_in_8bit,
        trust_remote_code=True,
    )
    # forward時にhiddenを返すよう設定
    model.config.output_hidden_states = True
    model.eval()

    print(f"✅ Model {model_name} loaded successfully.")
    return model, tokenizer
