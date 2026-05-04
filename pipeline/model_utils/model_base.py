from abc import ABC, abstractmethod
import os
import sys
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, GenerationConfig
from tqdm import tqdm
from torch import Tensor
from jaxtyping import Int, Float

from pipeline.utils.hook_utils import add_hooks

# Repo-root device_utils shim so ModelBase subclasses can load models onto
# whichever device is available (including TPU/XLA, which does not support
# accelerate's device_map="auto").
_APRS_ROOT = os.path.abspath(
    os.path.join(os.path.dirname(__file__), os.pardir, os.pardir, os.pardir)
)
if _APRS_ROOT not in sys.path:
    sys.path.insert(0, _APRS_ROOT)
try:
    from device_utils import (  # type: ignore
        is_xla_available as _dev_is_xla,
        get_device as _dev_get_device,
    )
except Exception:
    def _dev_is_xla():
        return False
    def _dev_get_device():
        if torch.cuda.is_available():
            return torch.device("cuda")
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")


def load_pretrained_for_device(model_path, *, dtype=torch.bfloat16, **kwargs):
    """Load an HF causal LM in a device-aware way.

    On CUDA/MPS/CPU, delegates to `AutoModelForCausalLM.from_pretrained` with
    `device_map="auto"` (preserving prior behavior). On XLA, omits `device_map`
    entirely and then `.to(xla_device())`, because accelerate's device_map does
    not support XLA placement.

    Any extra keyword arguments are forwarded to `from_pretrained`.
    """
    load_kwargs = dict(torch_dtype=dtype)
    load_kwargs.update(kwargs)
    if _dev_is_xla():
        model = AutoModelForCausalLM.from_pretrained(model_path, **load_kwargs)
        model = model.to(_dev_get_device())
    else:
        model = AutoModelForCausalLM.from_pretrained(
            model_path, device_map="auto", **load_kwargs
        )
    return model

class ModelBase(ABC):
    def __init__(self, model_name_or_path: str):
        self.model_name_or_path = model_name_or_path
        self.model: AutoModelForCausalLM = self._load_model(model_name_or_path)
        self.tokenizer: AutoTokenizer = self._load_tokenizer(model_name_or_path)
        
        self.tokenize_instructions_fn = self._get_tokenize_instructions_fn()
        self.eoi_toks = self._get_eoi_toks()
        self.refusal_toks = self._get_refusal_toks()

        self.model_block_modules = self._get_model_block_modules()
        self.model_attn_modules = self._get_attn_modules()
        self.model_mlp_modules = self._get_mlp_modules()

    def del_model(self):
        if hasattr(self, 'model') and self.model is not None:
            del self.model

    @abstractmethod
    def _load_model(self, model_name_or_path: str) -> AutoModelForCausalLM:
        pass

    @abstractmethod
    def _load_tokenizer(self, model_name_or_path: str) -> AutoTokenizer:
        pass

    @abstractmethod
    def _get_tokenize_instructions_fn(self):
        pass

    @abstractmethod
    def _get_eoi_toks(self):
        pass

    @abstractmethod
    def _get_refusal_toks(self):
        pass

    @abstractmethod
    def _get_model_block_modules(self):
        pass

    @abstractmethod
    def _get_attn_modules(self):
        pass

    @abstractmethod
    def _get_mlp_modules(self):
        pass

    @abstractmethod
    def _get_orthogonalization_mod_fn(self, direction: Float[Tensor, "d_model"]):
        pass

    @abstractmethod
    def _get_act_add_mod_fn(self, direction: Float[Tensor, "d_model"], coeff: float, layer: int):
        pass

    def generate_completions(self, dataset, fwd_pre_hooks=[], fwd_hooks=[], batch_size=8, max_new_tokens=64):
        generation_config = GenerationConfig(max_new_tokens=max_new_tokens, do_sample=False)
        generation_config.pad_token_id = self.tokenizer.pad_token_id

        completions = []
        instructions = [x['instruction'] for x in dataset]
        categories = [x['category'] for x in dataset]

        for i in tqdm(range(0, len(dataset), batch_size)):
            tokenized_instructions = self.tokenize_instructions_fn(instructions=instructions[i:i + batch_size])

            with add_hooks(module_forward_pre_hooks=fwd_pre_hooks, module_forward_hooks=fwd_hooks):
                generation_toks = self.model.generate(
                    input_ids=tokenized_instructions.input_ids.to(self.model.get_input_embeddings().weight.device),
                    attention_mask=tokenized_instructions.attention_mask.to(self.model.get_input_embeddings().weight.device),
                    generation_config=generation_config,
                )

                generation_toks = generation_toks[:, tokenized_instructions.input_ids.shape[-1]:]

                for generation_idx, generation in enumerate(generation_toks):
                    completions.append({
                        'category': categories[i + generation_idx],
                        'prompt': instructions[i + generation_idx],
                        'response': self.tokenizer.decode(generation, skip_special_tokens=True).strip()
                    })

        return completions
