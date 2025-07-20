import torch.nn as nn
import torch

from dataclasses import dataclass, field

@dataclass
class DiffusionArguments:
    r"""
    Arguments of Diffusion Models.
    """
    diffusion_steps: int = field(
        default=64,
        metadata={"help": "timesteps of diffusion models."}
    )
    generation_batch_size: int =field(
        default=6,
    )
    decoding_strategy: str = field(
        default="stochastic0.5-linear",
        metadata={"help": "<topk_mode>-<schedule>"}
    )
    token_reweighting: bool = field(
        default=False,
        metadata={"help": "use focal loss for token-level reweighting"}
    )
    alpha: float = field(
        default=0.25,
        metadata={"help": "for focal loss"}
    )
    gamma: float = field(
        default=2,
        metadata={"help": "for focal loss"}
    )
    time_reweighting: str = field(
        default='original',
        metadata={"help": "use focal loss for token-level reweighting"}
    )
    topk_decoding: bool = field(
        default=False,
        metadata={"help": "use focal loss for token-level reweighting"}
    )
    is_train: bool = field(
        default=True,
        metadata={"help": "use focal loss for token-level reweighting"}
    )
    checkpoint_dir: str = field(
        default= '',
    )

    def __post_init__(self):
        pass

class DiffusionModel(nn.Module):
    """
    diffusion model
    """

    def __init__(
        self,
        model,
        config,
        diffusion_args
    ):
        super().__init__()

        self.model = model
        self.config = self.model.config
        self.embed_dim = self.config.hidden_size
        self.hidden_dim = self.config.hidden_size
        self.vocab_size = config.vocab_size
        self.embed_tokens = self.model.transformer.wte
        self.denoise_model = self.model.transformer # use inputs_embeds instead of input_ids in forward function
        for gpt2block in self.model.transformer.h:
            gpt2block.attn.bias.fill_(True)  # remove causal mask
        self.lm_head = self.model.lm_head
        self.diffusion_args = diffusion_args
        self.warnings_issued = {}

    def get_logits(self, hidden_repr):
        return self.lm_head(hidden_repr)

    def get_input_embeddings(self):
        return self.embed_tokens
    
    def get_embeds(self, input_ids):
        return self.embed_tokens(input_ids)
    
    def forward(self, input_ids, t=None, attention_mask=None):
        """
        denoise the input
        """
        x_embed = self.get_embeds(input_ids)
        
        if attention_mask is None:
            attention_mask = torch.ones_like(input_ids) 

        x = self.denoise_model(inputs_embeds = x_embed, attention_mask=attention_mask, return_dict = False)[0]

        logits = self.get_logits(x)

        return logits

    def add_model_tags(self, tags):
        pass
