class ExpertLinearTuner(nn.Module):

    def __init__(self,
                 in_features: int,
                 out_features: int,
                 bias: bool = True,
                 device=None,
                 dtype=None,
                 expert_in_dim=None,
                 expert_out_dim=None,
                 **kwargs) -> None:
        super().__init__()

        self.original_linear = nn.Linear(in_features, out_features, bias, device, dtype)
        self.expert_linear = nn.Linear(expert_in_dim, expert_out_dim, bias, device, dtype)
        
        self.bias = bias

        self.out_features = out_features
        self.expert_in_dim = expert_in_dim
        self.expert_out_dim = expert_out_dim
        print(f"ExpertLinearTuner expert_in_dim={self.expert_in_dim} | expert_out_dim={self.expert_out_dim}")

        self.reset_parameters()

    def reset_parameters(self):
        if hasattr(self, 'expert_linear'):
            # initialize A the same way as the default for nn.Linear and B to zero
            nn.init.xavier_normal_(self.expert_linear.weight)

    def forward(self, x, im_mask=None):

        if im_mask is not None:
            if torch.sum(im_mask) > 0:
                
                res = torch.zeros((*x.shape[:-1], self.out_features), dtype=x.dtype, device=x.device)
                res[~im_mask] = self.original_linear(x[~im_mask])
                res[..., :self.expert_out_dim][im_mask] = self.expert_linear(x[..., :self.expert_in_dim][im_mask])
                
            else:
                res = self.original_linear(x)
                part_x = x[:, :1]
                expert_hidden = part_x[:, :, :self.expert_in_dim]
                res[:, :1] += self.expert_linear(expert_hidden).mean() * 0
        return res


class Qwen2MLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.hidden_size = config.hidden_size
        self.intermediate_size = config.intermediate_size
        
        '''
        self.gate_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
        self.up_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
        self.down_proj = nn.Linear(self.intermediate_size, self.hidden_size, bias=False)
        '''
        
        if ADAPTER_MODE == 'plora':
            self.gate_proj = PLoRA(
                self.hidden_size,
                self.intermediate_size,
                bias=False,
                lora_r=PLORA_R,
                lora_alpha=PLORA_ALPHA,
                lora_len=576)
            self.up_proj = PLoRA(
                self.hidden_size,
                self.intermediate_size,
                bias=False,
                lora_r=PLORA_R,
                lora_alpha=PLORA_ALPHA,
                lora_len=576)
            self.down_proj = PLoRA(
                self.intermediate_size,
                self.hidden_size,
                bias=False,
                lora_r=PLORA_R,
                lora_alpha=PLORA_ALPHA,
                lora_len=576)
        elif ADAPTER_MODE == 'expert':
            self.gate_proj = ExpertLinearTuner(
                self.hidden_size,
                self.intermediate_size,
                bias=False,
                expert_in_dim=EXPERT_DIM,
                expert_out_dim=EXPERT_HIDDEN,
                )
            self.up_proj = ExpertLinearTuner(
                self.hidden_size,
                self.intermediate_size,
                bias=False,
                expert_in_dim=EXPERT_DIM,
                expert_out_dim=EXPERT_HIDDEN,
                )
            self.down_proj = ExpertLinearTuner(
                self.intermediate_size,
                self.hidden_size,
                bias=False,
                expert_in_dim=EXPERT_HIDDEN,
                expert_out_dim=EXPERT_DIM,
                )
        else:
            raise ValueError(f"Unexpected adapter mode: {ADAPTER_MODE}")
        
        self.act_fn = ACT2FN[config.hidden_act]


class Qwen2Attention(nn.Module):
    """
    Multi-headed attention from 'Attention Is All You Need' paper. Modified to use sliding window attention: Longformer
    and "Generating Long Sequences with Sparse Transformers".
    """

    def __init__(self, config: Qwen2Config, layer_idx: Optional[int] = None):
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx
        if layer_idx is None:
            logger.warning_once(
                f"Instantiating {self.__class__.__name__} without passing `layer_idx` is not recommended and will "
                "to errors during the forward call, if caching is used. Please make sure to provide a `layer_idx` "
                "when creating this class."
            )

        self.hidden_size = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.head_dim = self.hidden_size // self.num_heads
        self.num_key_value_heads = config.num_key_value_heads
        self.num_key_value_groups = self.num_heads // self.num_key_value_heads
        self.max_position_embeddings = config.max_position_embeddings
        self.rope_theta = config.rope_theta
        self.is_causal = True
        self.attention_dropout = config.attention_dropout

        if (self.head_dim * self.num_heads) != self.hidden_size:
            raise ValueError(
                f"hidden_size must be divisible by num_heads (got `hidden_size`: {self.hidden_size}"
                f" and `num_heads`: {self.num_heads})."
            )
            
        '''  
        self.q_proj = nn.Linear(self.hidden_size, self.num_heads * self.head_dim, bias=True)
        self.k_proj = nn.Linear(self.hidden_size, self.num_key_value_heads * self.head_dim, bias=True)
        self.v_proj = nn.Linear(self.hidden_size, self.num_key_value_heads * self.head_dim, bias=True)
        self.o_proj = nn.Linear(self.num_heads * self.head_dim, self.hidden_size, bias=False)
        '''  
        
        if ADAPTER_MODE == 'expert':
            self.q_proj = ExpertLinearTuner(
                self.hidden_size,
                self.num_heads * self.head_dim,
                bias=True,
                expert_in_dim=EXPERT_DIM,
                expert_out_dim=self.num_heads * self.head_dim)
            
            self.k_proj = ExpertLinearTuner(
                self.hidden_size,
                self.num_key_value_heads * self.head_dim,
                bias=True,
                expert_in_dim=EXPERT_DIM,
                expert_out_dim=self.num_key_value_heads * self.head_dim)
            
            self.v_proj = ExpertLinearTuner(
                self.hidden_size,
                self.num_key_value_heads * self.head_dim,
                bias=True,
                expert_in_dim=EXPERT_DIM,
                expert_out_dim=self.num_key_value_heads * self.head_dim)
            
            self.o_proj = ExpertLinearTuner(
                self.num_heads * self.head_dim,
                self.hidden_size,
                bias=False,
                expert_in_dim=self.num_heads * self.head_dim,
                expert_out_dim=EXPERT_DIM)
                
        else:
            raise NotImplementedError
        
        self.rotary_emb = Qwen2RotaryEmbedding(
            self.head_dim,
            max_position_embeddings=self.max_position_embeddings,
            base=self.rope_theta,
        )


# transfer the ori weights to new keys 'original_linear'
# -----------------------------------------------------
import torch
from collections import OrderedDict
from safetensors.torch import load_file

model = load_file('./temp/ori_model.safetensors')
new_state_dict = OrderedDict()
lora_layer = ['gate_proj', 'up_proj', 'down_proj', 'q_proj', 'k_proj', 'v_proj', 'o_proj']

for name, param in model.items():
    is_lora = False
    is_embedding = False
    
    for lora_name in lora_layer:
        if lora_name in name:
            new_name = f"{lora_name}.original_linear".join(name.split(lora_name))
            new_state_dict[new_name] = param
            is_lora = True
            break
    
    if 'embed_tokens' in name:
        new_name = 'tok_embeddings'.join(name.split('embed_tokens'))
        is_embedding = True
        new_state_dict[new_name] = param
    
    if not is_lora and not is_embedding:
        new_state_dict[name] = param

import pdb; pdb.set_trace()
torch.save(new_state_dict, 'pytorch_model.bin')
