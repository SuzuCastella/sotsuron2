# sipit/models/hf_loader.py
import torch
from typing import Optional, Literal, Tuple
from transformers import AutoModelForCausalLM, AutoTokenizer

QuantizationMode = Literal["none", "int8", "int4", "gptq", "awq"]

def _to_torch_dtype(dtype_str: str) -> torch.dtype:
    table = {
        "float32": torch.float32,
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
    }
    if dtype_str not in table:
        raise ValueError(f"Unsupported dtype: {dtype_str}")
    return table[dtype_str]


def load_model_and_tokenizer(
    model_name: str,
    device: str = "cuda",
    dtype: str = "float32",
    quantization: QuantizationMode = "none",
    bnb_compute_dtype: str = "float16",   # 4bit/8bitの計算dtype
    bnb_quant_type: str = "nf4",          # 4bit量子化子（nf4/ fp4）
    trust_remote_code: bool = True,
) -> Tuple[AutoModelForCausalLM, AutoTokenizer]:
    """
    量子化オプションに応じて CausalLM をロードする。
    quantization:
      - "none" : 通常ロード（dtype指定有効）
      - "int8" : bitsandbytes 8bit
      - "int4" : bitsandbytes 4bit（bnb_quant_type, bnb_compute_dtype 指定可）
      - "gptq" : GPTQ 事前量子化モデル（モデル側が対応している前提）
      - "awq"  : AWQ 事前量子化モデル（モデル側が対応している前提）
    """
    print(f"🔹 Loading model: {model_name} on {device} (dtype={dtype}, quantization={quantization})")

    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=trust_remote_code)
    # GPT-2 系などは pad_token が無いことが多いので安全に補完
    if tokenizer.pad_token is None and tokenizer.eos_token is not None:
        tokenizer.pad_token = tokenizer.eos_token

    # device_map は GPU のとき "auto" にしておくと分散/オフロードが効く
    device_map = "auto" if device.startswith("cuda") else None

    if quantization == "none":
        torch_dtype = _to_torch_dtype(dtype)
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            dtype=torch_dtype,              # ← こちらを使うと新しいtransformersで警告が出ない
            device_map=device_map,
            trust_remote_code=trust_remote_code,
        )
        qdesc = f"none (dtype={torch_dtype})"

    elif quantization == "int8":
        # bitsandbytes 8bit
        # dtypeはここでは直接指定しない（bnbの計算dtypeに委譲）
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            load_in_8bit=True,
            device_map=device_map,
            trust_remote_code=trust_remote_code,
        )
        qdesc = "bitsandbytes int8"

    elif quantization == "int4":
        # bitsandbytes 4bit
        # 追加の引数は bnb 系の名前に合わせる
        _compute_dtype = _to_torch_dtype(bnb_compute_dtype)
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            load_in_4bit=True,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type=bnb_quant_type,   # "nf4" or "fp4"
            bnb_4bit_compute_dtype=_compute_dtype,
            device_map=device_map,
            trust_remote_code=trust_remote_code,
        )
        qdesc = f"bitsandbytes int4 (quant={bnb_quant_type}, compute={_compute_dtype})"

    elif quantization in ("gptq", "awq"):
        # 事前量子化済みモデルを前提として通常ロード
        # 多くのリポで dtype 指定は不要/無視されることがあるので付けない
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            device_map=device_map,
            trust_remote_code=trust_remote_code,
        )
        qdesc = quantization

    else:
        raise ValueError(f"Unsupported quantization mode: {quantization}")

    # 隠れ状態を forward 時に返す
    if hasattr(model, "config"):
        model.config.output_hidden_states = True
    model.eval()

    print(f"✅ Model loaded with quantization: {qdesc}")
    return model, tokenizer
