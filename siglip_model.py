import torch 

import torch.nn as nn 
from typing import Tuple
from dataclasses import dataclass
import math 
import torch.nn.functional as F 

@dataclass
class SiglipVisionConfig:
    hidden_size  : int = 768
    intermediate_size  : int = hidden_size * 4 
    num_hidden_layers : int = 12
    num_attention_heads : int = 12 
    num_channels      : int = 3
    image_size        : int = 224
    patch_size        : int = 16
    layer_norm_eps    : float = 1e-6
    attension_dropout : float  = 0.0 
    num_image_tokens  : int   = None 

class SiglipMLP(nn.Module):
    def __init__(self, config : SiglipVisionConfig):
        super().__init__(config)

        self.fc1    = nn.Linear(in_features=config.hidden_size, out_features= config.intermediate_size)
        self.gelu   = nn.GELU(approximate='tanh')
        self.fc2    = nn.Linear(in_features=config.intermediate_size, out_features=config.hidden_size)

    def forward(self, hidden_states ): 
        hidden_states = self.fc1(hidden_states)
        hidden_states = self.gelu(hidden_states)
        hidden_states = self.fc2(hidden_states)




class SiglipAttention(nn.Module):

    def __init__(self, config : SiglipVisionConfig):
        super().__init__()
        self.training = False 
        self.config = config
        self.c_attn  = nn.Linear(in_features=config.hidden_size, out_features=config.hidden_size *3)
        self.c_proj  = nn.Linear(in_features=config.hidden_size, out_features=config.hidden_size)


    def forward(self, hidden_states : torch.tensor):
        qkv    = self.c_attn(hidden_states) # (B,T, dmodel) -> (B,T, dmodel *3)
        q,k,v = qkv.split(self.config.hidden_size, dim=2)  # (B,T, dmodel*3) -> [(B,T dmodel)] *3 
        (B,P,d_model) = q.size()
        h = self.config.num_attention_heads

        d = d_model//h
        q = q.view(B,P,h, d ).transpose(1,2)
        k = k.view(B,P,h, d ).transpose(1,2)
        v = v.view(B,P,h, d ).transpose(1,2)
        att = q @ k.transpose(-1,-2) * (1/math.sqrt(d))  # (B,h,T,d) @ (B,h,d,T) -> (B,h,T,T)
        att_weights = F.softmax(att, dim=-1) 
        att_weights = F.dropout(att, p = self.dropout, training = self.training)
        out = att_weights @ v   # (B, h, T,T) @ (B,h, T, d)-> (B,h, T, d)
        out = out.transpose(1,2).contiguous().view(B,T,d_model) #(B,h, T,  d) -> (B, T, d_model)
        out = self.c_proj(out)
        return out, att_weights

class SiglipEncoderLayer(nn.Module):
    def __init__(self, config : SiglipVisionConfig):
        super().__init__()
        self.n_embd = config.hidden_size
        self.layer_norm1   = nn.LayerNorm(self.n_embd, eps=self.config.layer_norm_eps)
        self.attn          = SiglipAttention()
        self.layer_norm2   = nn.LayerNorm(config.n_embd)
        self.mlp           = SiglipMLP(config)

    def forward(self,hidden_states):
        hidden_states = hidden_states + self.attn(self.layer_norm1(hidden_states))
        hidden_states = hidden_states + self.mlp(self.layer_norm2(hidden_states))
    
        return hidden_states

class SiglipEncoder(nn.Module):
    def __init__(self, config : SiglipVisionConfig):
        super().__init__()
        self.config     = config
        self.encoder = nn.ModuleList([ SiglipEncoderLayer(config) for _ in range(self.config.num_hidden_layers)])

    def forward(self, hidden_states):
        for block in self.encoder:
            hidden_states = block(hidden_states)
        
        return hidden_states


class SiglipVisionEmbedding(nn.Module):
    def __init__(self, config : SiglipVisionConfig):
        super().__init__()
        self.config     = config
        # (B, C, H,W) -> (B, P, n_embd)
        self.patch_embedding = nn.Conv2d(
                                    in_channels=config.num_channels,
                                    out_channels= config.hidden_size, 
                                     kernel_size= config.patch_size,
                                    stride= config.patch_size,
                                    padding = "valid"
                                     )
        
        self.num_patches = (self.config.image_size // (self.config.patch_size **2))
        self.position_embedding = nn.Embedding(self.num_patches, self.config.hidden_size)

        self.register_buffer("position_ids", torch.arange(self.num_patches).expand((1,-1)), persistent=False)
        

    def forward(self,pixel_values):
        B,C,H,W = pixel_values.size()
        # (B,C,H,W) -> (B, n_embd, num_patches_H, num_patches_W) -> (B, n_embd, num_patches) -> (B, num_patches, n_embd)
        patch_embeds = self.patch_embedding(pixel_values)
        embedding    = patch_embeds.flatten(2).transpose(1,2)
        # (B, num_patches, n_embd) + (B, num_patches, n_embd) => (B, num_patches, n_embd)
        embedding    = embedding + self.position_embedding(self.position_ids) 
        return embedding
 
class SiglipVisionTransformer(nn.Module):
    def __init__(self, config : SiglipVisionConfig):
        super().__init__()
        self.config     = config

        self.embedding  = SiglipVisionEmbedding(config)
        self.encoder    = SiglipEncoder(config)
        self.post_layernorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
    
    def forward(self, pixel_values):
         # (B, C, H, W) -> (B, P, n_embd)
         hidden_states = self.embedding(pixel_values)
         last_hidden_states  =  self.encoder(hidden_states)
         last_hidden_states  = self.post_layernorm(last_hidden_states)

         return last_hidden_states
    

class SiglipVisionModel(nn.Module):

    def __init__(self, config : SiglipVisionConfig):
        super().__init__()
        self.config = config
        self.vision_model = SiglipVisionTransformer(config)

    def forward(self, pixel_values) -> Tuple:
        # (B, C, H, W) -> (B, P, n_embd)
        return self.vision_model(pixel_values)






