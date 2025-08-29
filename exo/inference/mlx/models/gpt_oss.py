from dataclasses import dataclass, field

import mlx.core as mx
import mlx.nn as nn

from mlx_lm.models.base import create_attention_mask
from mlx_lm.models.gpt_oss import TransformerBlock, ModelArgs as _ModelArgs

from ...shard import Shard
from .base import IdentityBlock


@dataclass
class ModelArgs(_ModelArgs):
  shard: Shard = field(default_factory=lambda: Shard("", 0, 0, 0))

  def __post_init__(self):
    # parent handles standard normalization
    try:
      super().__post_init__()
    except Exception:
      # Some mlx_lm versions may not define __post_init__
      pass

    if isinstance(self.shard, Shard):
      return
    if not isinstance(self.shard, dict):
      raise TypeError(f"Expected shard to be a Shard instance or a dict, got {type(self.shard)} instead")

    self.shard = Shard(**self.shard)


class GptOSSModel(nn.Module):
  def __init__(self, args: ModelArgs):
    super().__init__()
    self.args = args
    self.vocab_size = args.vocab_size
    self.num_hidden_layers = args.num_hidden_layers
    assert self.vocab_size > 0

    # Input/output embeddings depending on shard
    if self.args.shard.is_first_layer() or (self.args.shard.is_last_layer() and getattr(args, 'tie_word_embeddings', False)):
      self.embed_tokens = nn.Embedding(args.vocab_size, args.hidden_size)

    # Layer blocks (sharded)
    self.layers = []
    for i in range(self.num_hidden_layers):
      if self.args.shard.start_layer <= i <= self.args.shard.end_layer:
        self.layers.append(TransformerBlock(args=args))
      else:
        self.layers.append(IdentityBlock())

    # Final norm on the last shard
    if self.args.shard.is_last_layer():
      # GPT-OSS uses RMSNorm in the public configs
      eps = getattr(args, 'rms_norm_eps', 1e-5)
      self.norm = nn.RMSNorm(args.hidden_size, eps=eps)

  def __call__(
    self,
    inputs: mx.array,
    cache=None,
  ):
    if self.args.shard.is_first_layer():
      h = self.embed_tokens(inputs)
    else:
      h = inputs

    mask = None
    if h.ndim > 1 and h.shape[1] > 1:
      mask = create_attention_mask(h, cache)

    if cache is None:
      cache = [None]*len(self.layers)

    for layer, c in zip(self.layers, cache):
      h = layer(h, mask, cache=c)

    if self.args.shard.is_last_layer():
      h = self.norm(h)
    return h


class Model(nn.Module):
  def __init__(self, args: ModelArgs):
    super().__init__()
    self.args = args
    self.model_type = args.model_type
    self.model = GptOSSModel(args)
    if self.args.shard.is_last_layer():
      if not getattr(args, 'tie_word_embeddings', False):
        self.lm_head = nn.Linear(args.hidden_size, args.vocab_size, bias=False)

  def __call__(
    self,
    inputs: mx.array,
    cache=None,
  ):
    out = self.model(inputs, cache)
    if self.args.shard.is_last_layer():
      if getattr(self.args, 'tie_word_embeddings', False):
        out = self.model.embed_tokens.as_linear(out)
      else:
        out = self.lm_head(out)
    return out

  def sanitize(self, weights):
    # Keep only weights relevant to our shard and required heads/embeddings
    shard_state_dict = {}

    for key, value in weights.items():
      if "self_attn.rotary_emb.inv_freq" in key:
        continue
      if key.startswith('model.layers.'):
        try:
          layer_num = int(key.split('.')[2])
        except Exception:
          continue
        if self.args.shard.start_layer <= layer_num <= self.args.shard.end_layer:
          shard_state_dict[key] = value
      elif self.args.shard.is_first_layer() and key.startswith('model.embed_tokens'):
        shard_state_dict[key] = value
      elif (self.args.shard.is_last_layer() and getattr(self.args, 'tie_word_embeddings', False)) and key.startswith('model.embed_tokens'):
        shard_state_dict[key] = value
      elif (self.args.shard.is_last_layer() and not getattr(self.args, 'tie_word_embeddings', False)) and key.startswith('lm_head'):
        shard_state_dict[key] = value
      elif self.args.shard.is_last_layer() and (key.startswith('model.norm')):
        shard_state_dict[key] = value

    # Clean incompatible leftovers when using tied embeddings
    if getattr(self.args, 'tie_word_embeddings', False):
      shard_state_dict.pop("lm_head.weight", None)

    return shard_state_dict

  @property
  def layers(self):
    return self.model.layers

  @property
  def head_dim(self):
    # Prefer explicit head_dim if present, otherwise fallback to hidden_size // num_attention_heads
    if hasattr(self.args, 'head_dim') and self.args.head_dim:
      return self.args.head_dim
    if hasattr(self.args, 'num_attention_heads') and self.args.num_attention_heads:
      return self.args.hidden_size // self.args.num_attention_heads
    return None

  @property
  def n_kv_heads(self):
    return getattr(self.args, 'num_key_value_heads', None)

