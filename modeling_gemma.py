import torch 

import torch.nn as nn 
from typing import Optional, Tuple, List, 

from torch.nn import CrossEntropyLoss
import torch.nn.functional as F 
import math 
from siglip_model import SiglipVisionConfig, SiglipVisionModel
from dataclasses import dataclass




class KVCache():
    def __init__(self):
        self.key_cache : List[torch.Tensor] = []
        self.value_cache : List[torch.Tensor] = []

    def num_items(self) -> int : 
        if len(self.key_cache) ==0:
            return 0 
        else:
            return self.key_cache[0].shape[-2]
        
    def update(self, 
               key_states : torch.Tensor,
               value_states : torch.Tensor,
               layer_idx : int):
        
        if len(self.key_cache) <= layer_idx:
            self.key_cache.append(key_states)
            self.value_cache.append(value_states)
        else:
            self.key_cache[layer_idx] = torch.cat([self.key_cache[layer_idx],key_states],dim=-2)
            self.value_cache[layer_idx] = torch.cat([self.value_cache[layer_idx],value_states],dim=-2)
    
        return self.key_cache[layer_idx], self.value_cache[layer_idx]

@dataclass
class GemmaConfig:
    vocab_size : int 
    hidden_size : int 
    hidden_size : int 
    intermediate_size : int 
    num_hidden_layers : int 
    num_attention_heads : int
    num_key_value_heads : int
    head_dim   : int = 256 
    max_position_embedding : int = 8192
    rms_norm_eps : int = 1e-6 
    rope_theta : float = 10000.0
    attention_bias : bool = False
    attention_dropout : float  = 0.0 
    pad_token_id : int = None 


@dataclass
class PaliGemmaConfig : 
    vision_config : SiglipVisionConfig = None 
    text_config   : GemmaConfig = None 
    ignore_index  : int = -100 
    image_token_index : int = 256000
    vocab_size :int = 257152
    projection_dim :int = 2048
    pad_token_id : int = None 


class PaliGemmaMultiModalProjector(nn.Module):
    def __init__(self,config : PaliGemmaConfig):
        super().__init__()
        self.linear = nn.Linear(config.vision_config.hidden_size, config.vision_config.projection_dim, bias=True)
    
    def forward(self, image_features):
        hidden_states = self.linear(image_features)
        return hidden_states


class GemmaRMSNorm(nn.Module):

    def __init__(self, dim :int, eps : float = 1e-6):
        super().__init__()
        self.eps = eps 
        self.weight = nn.Parameter(torch.zeros(dim))
    
    def _norm(self, x):
        return x * torch.rsqrt(x.pow(2).mean(-1,keepdim=True)+self.eps)
    
    def forward(self,x):
        output = self._norm(x.float())
        output = output * (1.0 + self.weight.float())
        return output.type_as(x) 
    

class GemmaMLP(nn.Module):
    def __init__(self,config):
        super().__init__()
        self.config = config 
        self.hidden_size = config.hidden_size 
        self.intermediate_size = config.intermediate_size
        self.gate_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
        self.up_proj   = nn.Linear(self.hidden_size, self.intermediate_size, bias= False)
        self.down_proj = nn.Linear(self.intermediate_size, self.hidden_size, bias=False)


    def forward(self,x):
        return self.down_proj(nn.functional.gelu(self.gate_proj(x),approximate="tanh") * self.up_proj(x))
    
class GemmaAttention(nn.Module):

    def __init__(self, config : GemmaConfig, layer_idx : Optional[int] = None):
        super().__init__()
        self.config = config 
        self.layer_idx = layer_idx 

        self.attention_dropout = config.attention_dropout
        self.hidden_size       = config.hidden_size
        self.num_heads         = config.num_attention_heads
        self.head_dim          = config.head_dim 
        self.num_key_value_heads = config.num_key_value_heads
        self.num_key_value_groups = self.num_heads // self.num_key_value_heads
        self.max_position_embeddings = config.max_position_embedding    
        self.rope_theta  = config.rope_theta
        self.is_causal = True 

        # Wq : [1024, 8 * 128] = [1024, 1024]
        # Wk : [1024, 1 * 128] = [1024, 128]
        # Wv : [1024, 1 * 128] = [1024, 128] 
        self.q_proj  = nn.Linear(self.hidden_size, self.num_heads * self.head_dim , bias = config.attention_bias)
        self.k_proj  = nn.Linear(self.hidden_size, self.num_key_value_heads * self.head_dim, bias=config.attention_bias)
        self.v_proj  = nn.Linear(self.hidden_size, self.num_key_value_heads * self.head_dim, bias=config.attention_bias)
        self.o_proj  = nn.Linear(self.num_heads * self.head_dim, self.hidden_size, bias= config.attention_bias)


    def forward(self, hidden_states : torch.Tensor, 
                attention_mask : Optional[torch.Tensor] = None, 
                position_ids : Optional[torch.LongTensor] = None, 
                kv_cache : Optional[KVCache] = None):
        
        bsz, q_len, _ = hidden_states.size()
        query_states  = self.q_proj(hidden_states)
        key_states    = self.k_proj(hidden_states)
        value_states  = self.v_proj(hidden_states)
        query_states  = query_states.view(bsz,q_len, self.num_heads, self.head_dim).transpose(1,2)
        key_states    = key_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1,2)
        value_states  = value_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1,2)


        cos, sin = self.rotary_emb(value_states, position_ids, seq_len=None)
        query_states, key_states = apply_rotary_pos_emb(query_states,key_states, cos, sin)

        if kv_cache is not None:
            key_states, value_states = kv_cache.update(key_states, value_states, self.layer_idx)

        key_states   = repeat_kv(key_states, self.num_key_value_groups)
        value_states = repeat_kv(value_states, self.num_key_value_groups)


        attn_weights = torch.matmul(query_states, key_states.transpose(2,3))/ math.sqrt(self.head_dim)

        attn_weights = attn_weights + attention_mask
        
        attn_weights = F.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query_states.dtype)

        attn_weights = F.dropout(attn_weights, p=self.attention_dropout, training=self.training)
        attn_output  = torch.matmul(attn_weights, value_states)
         
        if attn_output.size() != (bsz, self.num_heads, q_len, self.head_dim):
            raise ValueError()
        
        attn_output = attn_output.transpose(1,2).contiguous()
        attn_output = attn_output.view(bsz, q_len, -1)
        attn_output = self.o_proj(attn_output)

        return attn_output, attn_weights
    
      
def repeat_kv(hidden_states : torch.Tensor, n_rep : int) -> torch.Tensor:
    batch, num_key_value_heads, slen, head_dim = hidden_states.shape

    if n_rep == 1:
        return hidden_states
    hidden_states = hidden_states[:, :, None, : , : ].expand(batch, num_key_value_heads, n_rep, slen, head_dim)
    return hidden_states.reshape(batch, num_key_value_heads * n_rep, slen, head_dim)

 

class GemmaDecoderLayer(nn.Module):
    def __init__(self, config : GemmaConfig):
        super().__init__()

        self.hidden_size = config.hidden_size
        self.self_attn = GemmaAttention(config=config, layer_idx = layer_idx)
        self.mlp = GemmaMLP(config)

        self.input_layernorm = GemmaRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attn_layernorm = GemmaRMSNorm(config.hidden_size,eps=config.rms_norm_eps)


class GemmaModel(nn.Module):
    def __init__(self, config : GemmaConfig):
        super().__init__()

        self.config = config 
        self.padding_idx = config.pad_token_id
        self.vocab_size = config.vocab_size

        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size, self.padding_idx)

        self.layers = nn.ModuleList([GemmaDecoderLayer(config,layer_idx) for layer_idx in range(config.num_hidden_layers)])
        self.norm   = GemmaRMSNorm(config.hidden_size, eps = config.rms_norm_eps)


    def get_input_embedding(self):
        return self.embed_tokens
    
    def forward(self, attention_mask : Optional[torch.Tensor] = None, 
                position_ids : Optional[torch.LongTensor] = None, 
                inputs_embeds : Optional[torch.FloatTensor] = None, 
                kv_cache : Optional[KVCache]= None):
        
        hidden_states = inputs_embeds
        normalizer = torch.tensor(self.config.hidden_size**0.5, dtype=hidden_states.dtype)
        hidden_states = hidden_states * normalizer 

        for decoder_layer in self.layers:
            hidden_states = decoder_layer(
                hidden_states,
                attention_mask=attention_mask,
                position_ids = position_ids,
                kv_cache=kv_cache
            )

        hidden_states = self.norm(hidden_states)






class GemmaForCausualLM(nn.Module):
    def __init__(self, config : PaliGemmaConfig):
        super().__init__()

        self.config = config 
        self.model  = GemmaModel(config)
        self.vocab_size = config.vocab_size

        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size,bias=False)

    def tie_weight(self):
        self.lm_head.weight = self.model.embed_token_weight 

    def forward( self, attention_mask : Optional[torch.Tensor] = None, 
                 postion_ids : Optional[torch.LongTensor] = None, 
                 inputs_embeds : Optional[torch.FloatTensor] =  None, 
                 kv_cache   : Optional[KVCache] = None, ) -> Tuple : 
        
        outputs = self.model(attention_mask= attention_mask,postion_ids = postion_ids, inputs_embeds= inputs_embeds, kv_cache = kv_cache)

        hidden_states = outputs 
        logits = self.lm_head(hidden_states)
        logits = logits.float() 
        return_data = {"logits":logits}

        if kv_cache is not None:
            return_data["kv_cache"] = kv_cache

        return return_data








class PaliGemmaForConditionalGeneration(nn.Module):

    def __init__(self, config : PaliGemmaConfig):
        super().__init__()
        self.config = config 

        self.vision_tower : SiglipVisionModel = SiglipVisionModel(config.vision_config)
        self.muti_modal_projector = PaliGemmaMultiModalProjector(config)
        self.vocab_size = config.vocab_size

        self.language_model  = GemmaForCausualLM(config.text_config)
        
        self.pad_token_id = self.config.token_id if self.config.pad_token_id is not None else -1 


    def tie_weights(self):
        return self.language_model.tie_weights()
    


    def _merge_input_ids_with_image_features(self, image_features, inputs_embeds, input_ids,attention_mask, kv_cache):

        _, _ , embed_dim = image_features.shape 
        batch_size, seqence_length  = input_ids.shape 
        dtype, device = inputs_embeds.dtype, inputs_embeds.device

        scaled_image_features = image_features / (self.config.hidden_size ** 0.5) 
        final_embedding = torch.zeros(batch_size, seqence_length, embed_dim, dtype=inputs_embeds.dtype )

        #Masking the <img>, <prompt-text>, <padding-mask> 
        text_mask   = (input_ids!=self.config.image_token_index) & (input_ids != self.pad_token_id)
        image_mask  = input_ids = self.config.image_token_index
        pad_mask    = input_ids == self.pad_token_id

        text_mask_expanded  = text_mask.unsqueeze(-1).expand(-1, -1, embed_dim)
        pad_mask_expanded   = pad_mask.unsqueeze(-1).expand(-1,-1, embed_dim)
        image_mask_expanded = image_mask.unsqueeze(-1).expand(-1,-1, embed_dim)

        final_embedding = torch.where(text_mask_expanded, inputs_embeds, final_embedding)
        final_embedding = final_embedding.masked_scatter(image_mask_expanded, scaled_image_features)


        ### CREATE THE ATTENTION MASK #### 

        dtype, device = inputs_embeds.dtype, inputs_embeds.device 
        min_dtype = torch.finfo(dtype).min 
        q_len  = inputs_embeds.shape[1]

        if kv_cache is None or kv_cache.num_items() == 0: 
            # First time creating a KV-Cache 
            causal_mask = torch.full((batch_size, q_len, q_len), fill_value=0, dtype=dtype, device=device)
        else:
            # At the time of inference 
            assert q_len == 1
            kv_len = kv_cache.num_items() + q_len 
            casual_mask = torch.full((batch_size, q_len, kv_len), fill_value=0, dtype=dtype, device=device)

        casual_mask = causal_mask.unsqueeze(1)

        if kv_cache is not None and kv_cache.num_items() >0 :
            position_ids = attention_mask.cumsum(-1)[:, -1]
            if position_ids.dim() == 1 : 
                position_ids = position_ids.unsqueeze(0)
            else:
                position_ids = (attention_mask.cumsum(-1)).masked_fill_((attention_mask==0), 1).to(device)
        

        return final_embedding, causal_mask, position_ids
    



    

     




    
    def forward(self, input_ids : torch.LongTensor = None, pixel_values : torch.FloatTensor = None, 
                attention_mask : Optional[torch.Tensor] = None, 
                kv_cache : Optional[KVCache] = None ) -> Tuple:
        
        inputs_embeds = self.language_model.get_input_embeddings()(input_ids)

        selected_image_feature = self.vision_tower(pixel_values.to(inputs_embeds.dtype))

        image_features =  self.muti_modal_projector(selected_image_feature)

        inputs_embeds, attention_mask, position_ids = self._merge_input_ids_with_image_features(image_features, inputs_embeds, input_ids,attention_mask, kv_cache)

        self.language_model(
            attention_mask = attention_mask,
            position_ids = position_ids,
            inputs_embds = inputs_embeds,
            kv_cache = kv_cache
        )


    



        
