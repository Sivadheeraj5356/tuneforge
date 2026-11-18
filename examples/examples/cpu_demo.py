import json
import torch
from pathlib import Path
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.pre_tokenizers import Whitespace
from transformers import Qwen3Config, Qwen3ForCausalLM, PreTrainedTokenizerFast
from tuneforge import TrainConfig
from tuneforge.train import train
from tuneforge.models import generate
from tuneforge.export import merge_adapter

root = Path('runs/cpu-demo')
root.mkdir(parents=True, exist_ok=True)
base = root / 'base'
torch.manual_seed(42)
vocab = {'[PAD]': 0, '[UNK]': 1, '[EOS]': 2, 'SELECT': 3, '1': 4, ';': 5, 'SQL': 6, ':': 7}
t = Tokenizer(WordLevel(vocab, unk_token='[UNK]'))
t.pre_tokenizer = Whitespace()
tok = PreTrainedTokenizerFast(tokenizer_object=t, pad_token='[PAD]', unk_token='[UNK]', eos_token='[EOS]')
tok.save_pretrained(base)
model = Qwen3ForCausalLM(Qwen3Config(vocab_size=8, hidden_size=32, intermediate_size=64, num_hidden_layers=1, num_attention_heads=2, num_key_value_heads=1, head_dim=16, max_position_embeddings=128, eos_token_id=2, pad_token_id=0))
model.save_pretrained(base)
rows = root / 'rows.jsonl'
rows.write_text('\n'.join(json.dumps({'prompt':'SQL : ', 'completion':'SELECT 1 ; [EOS]'}) for _ in range(8)))
cfg = TrainConfig(model=str(base), dataset='local:'+str(rows), output_dir=str(root/'adapter'), load_in_4bit=False, max_steps=2, per_device_batch_size=2, grad_accum=1, max_seq_len=64, logging_steps=1)
train(cfg)
print('GENERATED', repr(generate(str(base), 'SQL :', adapter=cfg.output_dir, max_new_tokens=4)))
print('MERGED', merge_adapter(str(base), cfg.output_dir, str(root/'merged')))
print('CPU demo complete. This is a tiny random model, not a quality benchmark.')
