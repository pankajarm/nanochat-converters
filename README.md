# nanochat-converters

Utilities to convert NanoChat checkpoints to Hugging Face format and GGUF.

## GGUF conversion (default: native nanochat)
```
python convert_and_upload_gguf.py \
  --source-repo pankajmathur/nanochat-d34-sft-hf \
  --arch nanochat \
  --base-dtype f16 \
  --skip-upload
```

Notes:
- `--arch nanochat` is now the default and matches the architecture (2-layer relu2 MLP, parameter-free RMSNorm, flipped RoPE, logit softcap). Requires llama.cpp with nanochat support.
- Tokenizer is padded to config vocab_size if HF tokenizer.json is short; placeholders are marked UNUSED.
- Validation fails fast on missing arch/vocab/block_count or mismatched embedding/lm_head shapes.

Fallbacks:
- `--arch gpt2`: structure matches (2-layer MLP) but activation is GELU.
- `--arch llama`: gated MLP with duplicated fc1; lowest quality. 

For best quality use `--arch nanochat` or the HF model directly.
