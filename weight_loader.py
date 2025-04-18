
import logging
logger = logging.getLogger("llama3_export")

def load_layer0_weights_into_custom_block(layer0, block):
    block.self_attn.q_proj.weight.data.copy_(layer0.self_attn.q_proj.weight.data.to_local())
    block.self_attn.k_proj.weight.data.copy_(layer0.self_attn.k_proj.weight.data.to_local())
    block.self_attn.v_proj.weight.data.copy_(layer0.self_attn.v_proj.weight.data.to_local())
    block.self_attn.o_proj.weight.data.copy_(layer0.self_attn.o_proj.weight.data.to_local())

    block.mlp.gate_proj.weight.data.copy_(layer0.mlp.gate_proj.weight.data.to_local())
    block.mlp.up_proj.weight.data.copy_(layer0.mlp.up_proj.weight.data.to_local())
    block.mlp.down_proj.weight.data.copy_(layer0.mlp.down_proj.weight.data.to_local())

    block.input_layernorm.weight.data.copy_(layer0.input_layernorm.weight.data.to_local())
    block.post_attention_layernorm.weight.data.copy_(layer0.post_attention_layernorm.weight.data.to_local())
