# sipit/core/policies/gradient.py
from __future__ import annotations
from dataclasses import dataclass
from typing import Callable, Iterable, Optional, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F

Tensor = torch.Tensor

@dataclass
class GradPolicyConfig:
    step_size: float = 1e-1        # γ
    steps: int = 1                 # 1でも論文のAlg3に忠実。必要なら増やす
    norm_clip: Optional[float] = None  # 勾配ノルムクリップ
    project_unit_ball: bool = False    # 埋め込みノルムを元語彙の中央値に射影したい時などに
    topk_return: int = 1           # 返す候補数（通常1）
    device: Optional[str] = None

class GradientPolicy:
    """
    Algorithm 3 (Gradient-based Policy)
    Ensure: next token id v* and its updated continuous embedding e^(j)

    必須入力:
      - E: 語彙埋め込み行列 [|V|, d]
      - e_init: 直前反復の連続埋め込み e^(j-1) [d]
      - target_h: 目標隠れ状態 ĥ_t [d_h]
      - F_embed: Callable(e: [d]) -> h_pred: [d_h]
          プレフィックス π・層 ℓ・時刻 t はクロージャ側で固定しておく
      - visited: 既訪問トークン集合 C
    """
    def __init__(self, cfg: GradPolicyConfig):
        self.cfg = cfg

    @torch.no_grad()
    def _rank_vocab_by_l2(self, E: Tensor, e: Tensor) -> Tensor:
        # returns indices sorted by ||E_v - e||_2 ascending
        # E:[V,d], e:[d]
        dists = torch.norm(E - e[None, :], dim=1)  # [V]
        return torch.argsort(dists, dim=0)

    def step(
        self,
        E: Tensor,                    # [V, d]
        e_init: Tensor,               # [d]
        target_h: Tensor,             # [d_h]
        F_embed: Callable[[Tensor], Tensor],
        visited: Iterable[int] = (),
    ) -> Tuple[int, Tensor]:
        device = self.cfg.device or (e_init.device if isinstance(e_init, torch.Tensor) else "cpu")
        E = E.to(device)
        target_h = target_h.to(device)

        # 1) 勾配計算対象の連続埋め込み e
        e = e_init.detach().to(device).clone().requires_grad_(True)

        # 2) j=1..steps まで更新（Alg.3の行1-2）
        opt = None  # 必要ならAdamに差し替え可
        for _ in range(max(1, self.cfg.steps)):
            h_pred = F_embed(e)                     # F(e; π,t)
            loss = 0.5 * F.mse_loss(h_pred, target_h, reduction="sum")  # (1/2)||...||^2
            (g,) = torch.autograd.grad(loss, (e,), retain_graph=False, create_graph=False)
            if self.cfg.norm_clip is not None:
                g = torch.clamp(g, -self.cfg.norm_clip, self.cfg.norm_clip)
            # 単純なGD更新: e <- e - γ g
            with torch.no_grad():
                e.add_(g, alpha=-self.cfg.step_size)
                if self.cfg.project_unit_ball:
                    # 任意の正則化（必要なら）：ノルム射影
                    e_norm = e.norm()
                    if e_norm > 1.0:
                        e.mul_(1.0 / e_norm)
                e.requires_grad_(True)

        e_final = e.detach()

        # 3) L を距離昇順で取得（Alg.3の行3）
        L = self._rank_vocab_by_l2(E, e_final)

        # 4) ρ(v;π) をランクと定義（実装上は単に順番）
        visited_set = set(int(v) for v in visited)

        # 5) 未訪問で最上位の v* を選択（Alg.3の行5）
        v_star = None
        for vid in L.tolist():
            if vid not in visited_set:
                v_star = vid
                break
        if v_star is None:
            # 念のためフォールバック（全訪問済みなら最上位）
            v_star = int(L[0].item())

        # 6) return v*, e^(j)（Alg.3の行6）
        return v_star, e_final
