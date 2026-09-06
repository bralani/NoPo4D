import torch
from torch import nn


def convert_to_buffer(module: nn.Module, persistent: bool = True):
    # Recurse over child modules.
    for name, child in list(module.named_children()):
        convert_to_buffer(child, persistent)

    # Also re-save buffers to change persistence.
    for name, parameter_or_buffer in (
        *module.named_parameters(recurse=False),
        *module.named_buffers(recurse=False),
    ):
        value = parameter_or_buffer.detach().clone()
        delattr(module, name)
        module.register_buffer(name, value, persistent=persistent)


def recursive_load_weights(
    model: nn.Module,
    state_dict_to_load: dict[str, torch.Tensor],
    prefix: str = "",
    strict_match: bool = True
) -> None:
    """
    Recursively loads weights into a model from a state dictionary, matching
    parameters based on their name and tensor shape.

    Weights are loaded only if both the name (key) and the shape of the tensor
    match between the model and the state_dict_to_load. This is useful for
    loading weights from a checkpoint with a slightly different structure or
    when the checkpoint is from a wrapped model.

    Args:
        model: The PyTorch model (nn.Module) to load weights into.
        state_dict_to_load: The state dictionary (from a checkpoint) to load weights from.
        prefix: Internal argument used for recursive calls to build the full parameter name.
        strict_match: If True, only load if both name and shape match.
                      If False, only load if shape matches (requires non-strict loading on nn.Module).
    """
    
    # Store keys that are successfully loaded to prevent trying to load them into submodules
    keys_loaded_in_this_level = set()

    # remove "model." from keys in state_dict_to_load
    state_dict_to_load = {k.replace("model.", "", 1): v for k, v in state_dict_to_load.items()}
    
    # 1. First, attempt to load weights directly into the parameters of the current module
    for name, param in model.named_parameters(recurse=False):
        full_name = f"{prefix}{name}".replace("model.", "", 1)
        
        if full_name in state_dict_to_load:
            checkpoint_tensor = state_dict_to_load[full_name]
            
            # Check for shape match
            if param.shape == checkpoint_tensor.shape:
                try:
                    # In-place copy the weight data
                    with torch.no_grad():
                        param.copy_(checkpoint_tensor)
                    print(f"Loaded: {full_name} (Shape: {tuple(param.shape)})")
                    keys_loaded_in_this_level.add(full_name)
                except Exception as e:
                    print(f"Failed to copy weight for {full_name}. Error: {e}")
            elif strict_match:
                print(f"Skipped: {full_name} due to shape mismatch. Model: {tuple(param.shape)}, Checkpoint: {tuple(checkpoint_tensor.shape)}")

    # 2. Recursively call for submodules
    for name, module in model.named_children():
        module_prefix = f"{prefix}{name}."

        # Filter the state_dict_to_load to only include keys that start with
        # the module's prefix and haven't been loaded by the parent module.
        sub_state_dict_to_load = {}
        for k, v in state_dict_to_load.items():
            if k.startswith(module_prefix) and k not in keys_loaded_in_this_level:
                sub_state_dict_to_load[k] = v

        recursive_load_weights(
            module,
            sub_state_dict_to_load,
            prefix=module_prefix,
            strict_match=strict_match
        )