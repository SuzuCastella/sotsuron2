# sipit/models/wrappers.py
import torch
from transformers.cache_utils import DynamicCache

class LMWrapper:
    """
    HFモデルをラップ。hidden/logits取得に加え、pastを用いた高速1ステップ推論、
    さらに「連続埋め込み e を直接差し込む forward」を提供。
    """
    def __init__(self, model, tokenizer):
        self.model = model
        self.tokenizer = tokenizer

    def encode(self, text: str):
        # GPT-2 等は add_special_tokens=False を明示
        return self.tokenizer(text, return_tensors="pt", add_special_tokens=False)

    @torch.inference_mode()
    def forward(self, input_ids, attention_mask=None, use_cache=False):
        return self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True,
            use_cache=use_cache,
            return_dict=True
        )

    # ---------- 高速化: KVキャッシュ ----------
    @torch.inference_mode()
    def build_past(self, prefix_ids: torch.LongTensor, attention_mask=None):
        """
        prefix を1回だけ流して past_key_values を得る。
        戻り値: (past_cache, past_length)
        past_cache は DynamicCache（新API）で返す。
        """
        device = next(self.model.parameters()).device
        out = self.model(
            input_ids=prefix_ids.to(device),
            attention_mask=(attention_mask.to(device) if attention_mask is not None else None),
            use_cache=True,
            output_hidden_states=True,
            return_dict=True
        )
        past_len = prefix_ids.shape[-1]
        past = out.past_key_values

        # 旧式(タプル) → 新仕様(DynamicCache) に正規化
        if past is not None and not hasattr(past, "get_seq_length"):
            past = DynamicCache.from_legacy_cache(past)

        return past, past_len

    def _repeat_past_for_batch(self, past, batch_size: int):
        """
        past（DynamicCache or legacy tuple or None）をバッチ B に複製して DynamicCache で返す。
        past が None の場合は None をそのまま返す（= キャッシュ無しで前進）。
        """
        if past is None:
            return None

        # 1) まず legacy 形式に落とす
        if hasattr(past, "to_legacy_cache"):   # DynamicCache
            legacy = past.to_legacy_cache()
        else:
            legacy = past  # すでに legacy tuple

        if batch_size == 1:
            # そのまま DynamicCache に戻して返す
            return DynamicCache.from_legacy_cache(legacy)

        # 2) legacy の各 (k,v) を B に expand
        expanded = []
        for (k, v) in legacy:
            # 形状: [B, num_heads, seq_len, head_dim]
            k_rep = k.expand(batch_size, -1, -1, -1).contiguous()
            v_rep = v.expand(batch_size, -1, -1, -1).contiguous()
            expanded.append((k_rep, v_rep))

        # 3) DynamicCache に戻す
        return DynamicCache.from_legacy_cache(tuple(expanded))

    @torch.inference_mode()
    def step_hidden_with_past(
        self,
        past,
        past_length: int,
        candidate_ids_batch: torch.LongTensor,   # [B, 1]
        layer_idx: int,
    ):
        """
        既存の past に対して、候補トークンのバッチを一括で1ステップ前進。
        past が None の場合は「キャッシュ無し1ステップ推論」に自動切替。
        戻り値: hidden_batch [B, hidden_size]
        """
        device = next(self.model.parameters()).device
        B = candidate_ids_batch.shape[0]

        if past is None:
            # === 初手（prefix が空）のケース: cache なしで素直に forward ===
            outputs = self.model(
                input_ids=candidate_ids_batch.to(device),  # [B,1]
                use_cache=False,
                output_hidden_states=True,
                return_dict=True
            )
            hs = outputs.hidden_states[layer_idx][:, -1, :]
            return hs

        # === 通常: cache ありで高速1ステップ ===
        past_batched = self._repeat_past_for_batch(past, B)  # None 以外 → DynamicCache
        position_ids = torch.full((B, 1), past_length, dtype=torch.long, device=device)

        outputs = self.model(
            input_ids=candidate_ids_batch.to(device),
            past_key_values=past_batched,
            position_ids=position_ids,
            use_cache=False,
            output_hidden_states=True,
            return_dict=True
        )
        hs = outputs.hidden_states[layer_idx][:, -1, :]
        return hs

    # ---------- 連続埋め込み e を直接差し込む ----------
    # 勾配が必要なので inference_mode にはしない
    def hidden_from_continuous_last(
        self,
        prefix_ids: torch.LongTensor,        # [1, L]  (Lは0でも可)
        e_last: torch.Tensor,                # [d_model]  ← requires_grad=True を想定
        layer_idx: int,
        attention_mask: torch.LongTensor = None,
    ) -> torch.Tensor:
        """
        prefix の token 埋め込みに、連続ベクトル e_last を末尾に連結して forward。
        戻り値: 層 layer_idx の「末尾位置」の隠れ状態 [d_model]
        """
        device = next(self.model.parameters()).device
        emb_layer = self.model.get_input_embeddings()  # nn.Embedding

        # 勾配有効
        with torch.enable_grad():
            if prefix_ids is not None and prefix_ids.numel() > 0:
                prefix_emb = emb_layer(prefix_ids.to(device))          # [1, L, d]
                e_last = e_last.to(device).unsqueeze(0).unsqueeze(0)  # [1,1,d]
                inputs_embeds = torch.cat([prefix_emb, e_last], dim=1) # [1, L+1, d]
                attn = (torch.cat([torch.ones_like(prefix_ids), torch.ones((1,1), dtype=torch.long, device=device)], dim=1)
                        if attention_mask is None else
                        torch.cat([attention_mask.to(device), torch.ones((1,1), dtype=torch.long, device=device)], dim=1))
            else:
                inputs_embeds = e_last.to(device).unsqueeze(0).unsqueeze(0)  # [1,1,d]
                attn = torch.ones((1,1), dtype=torch.long, device=device)

            outputs = self.model(
                inputs_embeds=inputs_embeds,
                attention_mask=attn,
                output_hidden_states=True,
                use_cache=False,
                return_dict=True
            )
            return outputs.hidden_states[layer_idx][0, -1, :]
