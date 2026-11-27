"""Write K/V into target slots and mask inactive slots."""

from vllm.model_executor.layers.attention.attention import unified_kv_cache_update


def backfill_kv(layer, key, value):
    key = key.view(-1, layer.num_kv_heads, layer.head_size)
    value = value.view(-1, layer.num_kv_heads, layer.head_size_v)
    unified_kv_cache_update(key, value, layer.layer_name)


def mask_slot_mapping(slot_mapping, active):
    """Share masked tensors across layer names attached to the same KV group."""
    by_tensor = {}
    result = {}
    for name, slots in slot_mapping.items():
        if id(slots) not in by_tensor:
            by_tensor[id(slots)] = slots.masked_fill(~active, -1)
        result[name] = by_tensor[id(slots)]
    return result
