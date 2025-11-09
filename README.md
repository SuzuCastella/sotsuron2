# 実行方法
## 仮想環境の構築
```
python -m venv venv
source venv/Scripts/activate
pip install -r requirements.txt
```

## デモの実行

```
python sipit/cli/invert_prompt.py   --model gpt2   --device cuda   --dtype float32   --text "Hello, world!"   --layer -1   --eps 0.003   --topk 0   --max_tokens 6
```

- --model: モデルの指定
- --device: GPU推奨
- --dtype:
- --text: サンプル文章
- --layer: 参照する中間層の指定
- --eps: L2距離の閾値
- --topk: 登録トークン農地参照するトークン数(0で全参照)
- --max_tokens: 何番目のトークンまで入力トークンを推定するか

実行成功例
```
 Loading model: gpt2 on cuda (float32)
`torch_dtype` is deprecated! Use `dtype` instead!
✅ Model gpt2 loaded successfully.
[t=1] matched token_id=15496 'Hello' (l2=1.5680e-04)
[t=2] matched token_id=11 ',' (l2=6.2680e-05)
[t=3] matched token_id=995 ' world' (l2=4.9465e-05)
[t=4] matched token_id=0 '!' (l2=0.0000e+00)

=== RESULT ===
True : Hello, world!
Recov: Hello, world!
Match: True
```

