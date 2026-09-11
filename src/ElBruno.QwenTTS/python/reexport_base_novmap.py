"""Re-export Base model LM ONNX models with vmap-free masking for C# ORT compatibility.

All dimensions are read from the model config, supporting both 0.6B and 1.7B variants.

Usage:
  python reexport_base_novmap.py --model-dir python/models/Qwen3-TTS-0.6B-Base --output-dir python/onnx_base
  python reexport_base_novmap.py --model-dir python/models/Qwen3-TTS-1.7B-Base --output-dir python/onnx_base_1.7b
"""

import argparse
import torch
import torch.nn as nn
import os
from transformers.masking_utils import ALL_MASK_ATTENTION_FUNCTIONS
from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS
from transformers.integrations.executorch import sdpa_mask_without_vmap
from transformers.cache_utils import DynamicCache

# Register vmap-free functions (ExecuTorch approach)
ALL_MASK_ATTENTION_FUNCTIONS.register('sdpa_without_vmap', sdpa_mask_without_vmap)
ALL_ATTENTION_FUNCTIONS.register('sdpa_without_vmap', ALL_ATTENTION_FUNCTIONS['sdpa'])

from qwen_tts.core.models.modeling_qwen3_tts import Qwen3TTSForConditionalGeneration
from qwen_tts.core.models.configuration_qwen3_tts import Qwen3TTSConfig

OPSET_VERSION = 17


def read_model_dims(config):
    """Extract export-relevant dimensions from model config."""
    tc = config.talker_config
    cp = tc.code_predictor_config
    dims = {
        "talker_num_layers": tc.num_hidden_layers,
        "talker_num_kv_heads": tc.num_key_value_heads,
        "talker_head_dim": getattr(tc, "head_dim", 128),
        "talker_hidden": tc.hidden_size,
        "talker_vocab": tc.vocab_size,
        "cp_num_layers": cp.num_hidden_layers,
        "cp_num_kv_heads": cp.num_key_value_heads,
        "cp_head_dim": getattr(cp, "head_dim", 128),
        "cp_hidden": cp.hidden_size,
        "cp_vocab": cp.vocab_size,
        "cp_num_groups": tc.num_code_groups - 1,
    }
    print(f"  Talker:  hidden={dims['talker_hidden']}, layers={dims['talker_num_layers']}, "
          f"kv_heads={dims['talker_num_kv_heads']}, head_dim={dims['talker_head_dim']}")
    print(f"  CodePred: hidden={dims['cp_hidden']}, layers={dims['cp_num_layers']}, "
          f"kv_heads={dims['cp_num_kv_heads']}, groups={dims['cp_num_groups']}")
    return dims


class TalkerPrefillWrapper(nn.Module):
    def __init__(self, talker, num_layers):
        super().__init__()
        self.model = talker.model
        self.codec_head = talker.codec_head
        self.num_layers = num_layers

    def forward(self, inputs_embeds, attention_mask, position_ids):
        outputs = self.model(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            position_ids=position_ids,
            use_cache=True,
        )
        hidden_states = outputs.last_hidden_state
        logits = self.codec_head(hidden_states[:, -1:, :])
        cache = outputs.past_key_values
        flat_kv = []
        for i in range(self.num_layers):
            flat_kv.append(cache.layers[i].keys)
            flat_kv.append(cache.layers[i].values)
        return (logits, hidden_states, *flat_kv)


class TalkerDecodeWrapper(nn.Module):
    def __init__(self, talker, num_layers):
        super().__init__()
        self.model = talker.model
        self.codec_head = talker.codec_head
        self.num_layers = num_layers

    def forward(self, inputs_embeds, attention_mask, position_ids, past_keys, past_values):
        cache = DynamicCache()
        for i in range(self.num_layers):
            cache.update(past_keys[i], past_values[i], i)
        outputs = self.model(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            position_ids=position_ids,
            use_cache=True,
            past_key_values=cache,
        )
        hidden_states = outputs.last_hidden_state
        logits = self.codec_head(hidden_states)
        new_cache = outputs.past_key_values
        present_keys = torch.stack([new_cache.layers[i].keys for i in range(self.num_layers)])
        present_values = torch.stack([new_cache.layers[i].values for i in range(self.num_layers)])
        return (logits, hidden_states, present_keys, present_values)


class CodePredictorWrapper(nn.Module):
    def __init__(self, code_predictor, num_layers):
        super().__init__()
        self.model = code_predictor.model
        # small_to_mtp_projection exported as .npy, not baked into ONNX
        self.num_layers = num_layers
        all_weights = torch.stack([head.weight for head in code_predictor.lm_head])
        self.register_buffer("lm_head_weights", all_weights)

    def forward(self, inputs_embeds, generation_steps, past_keys, past_values):
        # Projection applied externally by C# — not baked into ONNX
        cache = DynamicCache()
        for i in range(self.num_layers):
            cache.update(past_keys[i], past_values[i], i)
        outputs = self.model(
            inputs_embeds=inputs_embeds,
            use_cache=True,
            past_key_values=cache,
        )
        hidden_states = outputs.last_hidden_state
        weight = torch.index_select(self.lm_head_weights, 0, generation_steps.view(-1))
        weight = weight.squeeze(0)
        logits = torch.matmul(hidden_states, weight.t())
        new_cache = outputs.past_key_values
        present_keys = torch.stack([new_cache.layers[i].keys for i in range(self.num_layers)])
        present_values = torch.stack([new_cache.layers[i].values for i in range(self.num_layers)])
        return (logits, present_keys, present_values)


def patch_model_for_vmap_free(model):
    """Patch all sub-models to use vmap-free masking."""
    # Talker
    model.talker.model.config._attn_implementation = 'sdpa_without_vmap'
    for layer in model.talker.model.layers:
        layer.self_attn.config._attn_implementation = 'sdpa_without_vmap'
    # Code Predictor
    model.talker.code_predictor.model.config._attn_implementation = 'sdpa_without_vmap'
    for layer in model.talker.code_predictor.model.layers:
        layer.self_attn.config._attn_implementation = 'sdpa_without_vmap'


def fix_external_data_ref(onnx_path):
    """Ensure external data references use the correct filename."""
    import onnx
    model = onnx.load(onnx_path, load_external_data=False)
    expected_data = os.path.basename(onnx_path) + '.data'
    for tensor in model.graph.initializer:
        for entry in tensor.external_data:
            if entry.key == 'location':
                entry.value = expected_data
    onnx.save(model, onnx_path)


def export_prefill(talker, output_dir, dims):
    print("Exporting talker_prefill.onnx ...")
    wrapper = TalkerPrefillWrapper(talker, dims["talker_num_layers"]).eval()
    B, T = 1, 8
    dummy = (
        torch.randn(B, T, dims["talker_hidden"]),
        torch.ones(B, T, dtype=torch.int64),
        torch.zeros(3, B, T, dtype=torch.int64),
    )
    input_names = ['inputs_embeds', 'attention_mask', 'position_ids']
    output_names = ['logits', 'hidden_states']
    for i in range(dims["talker_num_layers"]):
        output_names.extend([f'present_key_{i}', f'present_value_{i}'])
    dynamic_axes = {
        'inputs_embeds': {0: 'batch_size', 1: 'sequence_length'},
        'attention_mask': {0: 'batch_size', 1: 'sequence_length'},
        'position_ids': {1: 'batch_size', 2: 'sequence_length'},
        'logits': {0: 'batch_size'},
        'hidden_states': {0: 'batch_size', 1: 'sequence_length'},
    }
    for i in range(dims["talker_num_layers"]):
        dynamic_axes[f'present_key_{i}'] = {0: 'batch_size', 2: 'sequence_length'}
        dynamic_axes[f'present_value_{i}'] = {0: 'batch_size', 2: 'sequence_length'}

    path = os.path.join(output_dir, 'talker_prefill.onnx')
    torch.onnx.export(wrapper, dummy, path,
        input_names=input_names, output_names=output_names,
        dynamic_axes=dynamic_axes, opset_version=OPSET_VERSION,
        do_constant_folding=True)
    fix_external_data_ref(path)
    print("  Done")


def export_decode(talker, output_dir, dims):
    print("Exporting talker_decode.onnx ...")
    wrapper = TalkerDecodeWrapper(talker, dims["talker_num_layers"]).eval()
    B, T_past = 1, 8
    dummy = (
        torch.randn(B, 1, dims["talker_hidden"]),
        torch.ones(B, T_past + 1, dtype=torch.int64),
        torch.zeros(3, B, 1, dtype=torch.int64),
        torch.randn(dims["talker_num_layers"], B, dims["talker_num_kv_heads"], T_past, dims["talker_head_dim"]),
        torch.randn(dims["talker_num_layers"], B, dims["talker_num_kv_heads"], T_past, dims["talker_head_dim"]),
    )
    input_names = ['inputs_embeds', 'attention_mask', 'position_ids', 'past_keys', 'past_values']
    output_names = ['logits', 'hidden_states', 'present_keys', 'present_values']
    dynamic_axes = {
        'inputs_embeds': {0: 'batch_size'},
        'attention_mask': {0: 'batch_size', 1: 'total_sequence_length'},
        'position_ids': {1: 'batch_size'},
        'logits': {0: 'batch_size'},
        'hidden_states': {0: 'batch_size'},
        'past_keys': {1: 'batch_size', 3: 'past_sequence_length'},
        'past_values': {1: 'batch_size', 3: 'past_sequence_length'},
        'present_keys': {1: 'batch_size', 3: 'total_sequence_length'},
        'present_values': {1: 'batch_size', 3: 'total_sequence_length'},
    }
    path = os.path.join(output_dir, 'talker_decode.onnx')
    torch.onnx.export(wrapper, dummy, path,
        input_names=input_names, output_names=output_names,
        dynamic_axes=dynamic_axes, opset_version=OPSET_VERSION,
        do_constant_folding=True)
    fix_external_data_ref(path)
    print("  Done")


def export_code_predictor(talker, output_dir, dims):
    print("Exporting code_predictor.onnx ...")
    wrapper = CodePredictorWrapper(talker.code_predictor, dims["cp_num_layers"]).eval()
    B, S, T_past = 1, 2, 2
    dummy = (
        torch.randn(B, S, dims["cp_hidden"]),
        torch.tensor([0], dtype=torch.int64),
        torch.randn(dims["cp_num_layers"], B, dims["cp_num_kv_heads"], T_past, dims["cp_head_dim"]),
        torch.randn(dims["cp_num_layers"], B, dims["cp_num_kv_heads"], T_past, dims["cp_head_dim"]),
    )
    input_names = ['inputs_embeds', 'generation_steps', 'past_keys', 'past_values']
    output_names = ['logits', 'present_keys', 'present_values']
    dynamic_axes = {
        'inputs_embeds': {0: 'batch_size', 1: 'sequence_length'},
        'generation_steps': {0: 'num_steps'},
        'past_keys': {1: 'batch_size', 3: 'past_sequence_length'},
        'past_values': {1: 'batch_size', 3: 'past_sequence_length'},
        'logits': {0: 'batch_size', 1: 'sequence_length'},
        'present_keys': {1: 'batch_size', 3: 'total_sequence_length'},
        'present_values': {1: 'batch_size', 3: 'total_sequence_length'},
    }
    path = os.path.join(output_dir, 'code_predictor.onnx')
    torch.onnx.export(wrapper, dummy, path,
        input_names=input_names, output_names=output_names,
        dynamic_axes=dynamic_axes, opset_version=OPSET_VERSION,
        do_constant_folding=True, dynamo=False)
    fix_external_data_ref(path)
    print("  Done")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description="Re-export Base model LM ONNX models with vmap-free masking"
    )
    parser.add_argument(
        '--model-dir', type=str,
        default='python/models/Qwen3-TTS-0.6B-Base',
        help='Path to the Qwen3-TTS Base model directory',
    )
    parser.add_argument(
        '--output-dir', type=str,
        default='python/onnx_base',
        help='Directory to save ONNX files',
    )
    args = parser.parse_args()

    print(f"Loading Base model from {args.model_dir}...")
    config = Qwen3TTSConfig.from_pretrained(args.model_dir)
    print("Model dimensions (from config):")
    dims = read_model_dims(config)

    model = Qwen3TTSForConditionalGeneration.from_pretrained(
        args.model_dir,
        dtype=torch.float32,
        attn_implementation='sdpa',
    )
    model.eval()
    patch_model_for_vmap_free(model)

    output_dir = args.output_dir
    os.makedirs(output_dir, exist_ok=True)

    with torch.no_grad():
        export_prefill(model.talker, output_dir, dims)
        export_decode(model.talker, output_dir, dims)
        export_code_predictor(model.talker, output_dir, dims)
    print("\nAll Base model ONNX files exported successfully!")
