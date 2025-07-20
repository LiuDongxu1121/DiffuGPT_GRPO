
import json
from pathlib import Path
import os
from typing import Dict, List, Sequence, Union
from typing import List
import torch

from transformers.tokenization_utils import AddedToken, PreTrainedTokenizer
from transformers import (
    AutoConfig,
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    PretrainedConfig,
    PreTrainedModel,
    PreTrainedTokenizerBase
)
from diffugpt_model import DiffusionModel, DiffusionArguments

def load_tokenizer(model_path):
    tokenizer = CustomTokenizer.from_pretrained(model_path)
    if isinstance(tokenizer, PreTrainedTokenizerBase) and "AutoTokenizer" in tokenizer.init_kwargs.get("auto_map", {}):
        tokenizer.__class__.register_for_auto_class()
    return tokenizer

def load_model(grpo_config):
    config_kwargs = {
            "trust_remote_code": True,
        }
    config = AutoConfig.from_pretrained(grpo_config.model_path, **config_kwargs)
    model = AutoModelForCausalLM.from_config(
        config,
    )
    diffu_args = DiffusionArguments(
        diffusion_steps = grpo_config.diffusion_steps,
        generation_batch_size = grpo_config.generation_batch_size,
        checkpoint_dir = './model_config/'
    )
    model = DiffusionModel(model, config, diffu_args)
    if isinstance(model, PreTrainedModel) and "AutoModelForCausalLM" in getattr(config, "auto_map", {}):
        model.__class__.register_for_auto_class()

    
    checkpoint_dir = diffu_args.checkpoint_dir
    is_trainable = diffu_args.is_train
    if checkpoint_dir is not None: # for sampling
        load_path = os.path.join(checkpoint_dir, 'pytorch_model.bin')
        loaded = torch.load(
            load_path,
            map_location=torch.device('cuda')
        )
        model.load_state_dict(loaded, strict=False)
        print(f"Loading pretrained model from {load_path}")
    model = model.train() if is_trainable else model.eval()
        
    # Prepare model for inference
    if not is_trainable:
        model.requires_grad_(False) # fix all model params
        model = model.to(model_args.compute_dtype) if model_args.quantization_bit is None else model

    trainable_params, all_param = count_parameters(model)
    print("trainable params: {:d} || all params: {:d} || trainable%: {:.4f}".format(
        trainable_params, all_param, 100 * trainable_params / all_param
    ))
    return model


class CustomTokenizer(PreTrainedTokenizer):
    model_input_names: List[str] = ["input_ids", "attention_mask"]

    def __init__(self, vocab: Sequence[str], model_max_length: int, **kwargs):
        """
        Args:
            vocab (Sequence[str]): List of desired tokens. Following are list of all of the special tokens with
                their corresponding ids:
                    "[PAD]": 0,
                    "[SEP]": 1,
                    "[MASK]": 2,
                    "[EOS]": 3,
                    "[UNK]": 4,
                an id (starting at 5) will be assigned to each character.

            model_max_length (int): Model maximum sequence length.
        """
        self.vocab = vocab
        self.model_max_length = model_max_length
        eos_token = AddedToken("[EOS]", lstrip=False, rstrip=False)
        sep_token = AddedToken("[SEP]", lstrip=False, rstrip=False)
        pad_token = AddedToken("[PAD]", lstrip=False, rstrip=False)
        unk_token = AddedToken("[UNK]", lstrip=False, rstrip=False)
        mask_token = AddedToken("[MASK]", lstrip=True, rstrip=False)

        self._vocab_str_to_int = {
            "[PAD]": 0,
            "[SEP]": 1,
            "[MASK]": 2,
            "[EOS]": 3,
            "[UNK]": 4,
            **{ch: i + 5 for i, ch in enumerate(vocab)},
        }
        self._vocab_int_to_str = {v: k for k, v in self._vocab_str_to_int.items()}

        super().__init__(
            eos_token=eos_token,
            sep_token=sep_token,
            pad_token=pad_token,
            mask_token=mask_token,
            unk_token=unk_token,
            add_prefix_space=False,
            model_max_length=model_max_length,
            **kwargs,
        )

    @property
    def vocab_size(self) -> int:
        return len(self._vocab_str_to_int)

    def _tokenize(self, text: str) -> List[str]:
        # suppose text is a like "split1 split2 split3", convert to character if split* not in vocab
        splits = text.split(' ')
        tokens = []
        for split in splits:
            if split is not '':
                if split in self._vocab_str_to_int:
                    tokens.extend([split, ' '])
                else:
                    tokens.extend(list(split) + [' '])
        return tokens[:-1]

    def _convert_token_to_id(self, token: str) -> int:
        return self._vocab_str_to_int.get(token, self._vocab_str_to_int["[UNK]"])

    def _convert_id_to_token(self, index: int) -> str:
        return self._vocab_int_to_str[index]

    def convert_tokens_to_string(self, tokens):
        return "".join(tokens)

    def get_config(self) -> Dict:
        return {
            "vocab": self.vocab,
            "model_max_length": self.model_max_length,
        }
    
    def get_vocab(self) -> Dict[str, int]:
        return self._vocab_str_to_int
    
    @classmethod
    def from_config(cls, config: Dict) -> "CustomTokenizer":
        cfg = {}
        cfg["vocab"] = config['vocab']
        cfg["model_max_length"] = config["model_max_length"]
        return cls(**cfg)

    def save_pretrained(self, save_directory: Union[str, os.PathLike], **kwargs):
        cfg_file = Path(save_directory) / "tokenizer_config.json"
        cfg = self.get_config()
        with open(cfg_file, "w") as f:
            json.dump(cfg, f, indent=4)

    @classmethod
    def from_pretrained(cls, save_directory: Union[str, os.PathLike], **kwargs):
        cfg_file = Path(save_directory) / "tokenizer_config.json"
        with open(cfg_file) as f:
            cfg = json.load(f)
        return cls.from_config(cfg)

    def __call__(
        self,
        text,
        return_tensors=None,
        padding=False,
        add_special_tokens=False,
        **kwargs
    ):
        if isinstance(text, str):
            texts = [text]
        else:
            texts = list(text)
        encodings = [self.encode(t, add_special_tokens=add_special_tokens) for t in texts]
        max_len = 64
        pad_id = self._vocab_str_to_int["[PAD]"]

        input_ids, attention_masks, src_masks = [], [], []
        for ids in encodings:
            ids = ids[:max_len] if len(ids) > max_len else ids + [pad_id] * (max_len - len(ids))
            mask = [1 if i != pad_id else 0 for i in ids]
            idx = ids.index(1)                  
            src_mask = [1] * (idx + 1) + [0] * (len(ids) - idx - 1)

            input_ids.append(ids)
            attention_masks.append(mask)
            src_masks.append(src_mask)
        if return_tensors == "pt":
            input_ids = torch.tensor(input_ids, dtype=torch.long)
            attention_masks = torch.tensor(attention_masks, dtype=torch.long)
            src_masks = torch.tensor(src_masks, dtype=torch.long)

        return {
            "input_ids": input_ids, 
            "attention_mask": attention_masks,
            "src_mask": src_masks,
            }


def count_parameters(model: torch.nn.Module) :
    r"""
    Returns the number of trainable parameters and number of all parameters in the model.
    """
    trainable_params, all_param = 0, 0
    for param in model.parameters():
        num_params = param.numel()
        # if using DS Zero 3 and the weights are initialized empty
        if num_params == 0 and hasattr(param, "ds_numel"):
            num_params = param.ds_numel

        # Due to the design of 4bit linear layers from bitsandbytes, multiply the number of parameters by 2
        if param.__class__.__name__ == "Params4bit":
            num_params = num_params * 2

        all_param += num_params
        if param.requires_grad:
            trainable_params += num_params

    return trainable_params, all_param

if __name__ == '__main__':
    tokenizer = CustomTokenizer.from_pretrained('./model_config')
    print(tokenizer.vocab_size)