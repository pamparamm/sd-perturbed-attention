from importlib import import_module


# Copied from https://github.com/comfyanonymous/ComfyUI/blob/719fb2c81d716ce8edd7f1bdc7804ae160a71d3a/comfy/model_patcher.py#L21 for backward compatibility
def set_model_options_patch_replace(model_options, patch, name, block_name, number, transformer_index=None):
    to = model_options["transformer_options"].copy()

    if "patches_replace" not in to:
        to["patches_replace"] = {}
    else:
        to["patches_replace"] = to["patches_replace"].copy()

    if name not in to["patches_replace"]:
        to["patches_replace"][name] = {}
    else:
        to["patches_replace"][name] = to["patches_replace"][name].copy()

    if transformer_index is not None:
        block = (block_name, number, transformer_index)
    else:
        block = (block_name, number)
    to["patches_replace"][name][block] = patch
    model_options["transformer_options"] = to
    return model_options


BACKEND = None

if not BACKEND:
    try:
        import comfy.model_patcher
        from comfy.model_patcher import ModelPatcher
        from comfy.samplers import calc_cond_batch

        set_model_options_patch_replace = getattr(comfy.model_patcher, "set_model_options_patch_replace", set_model_options_patch_replace)

        BACKEND = "ComfyUI"
    except ImportError as _:
        pass

if not BACKEND:
    try:
        from ldm_patched.modules.model_patcher import ModelPatcher
        from ldm_patched.modules.samplers import calc_cond_uncond_batch

        BACKEND = "Forge"
    except ImportError as _:
        pass

if not BACKEND:
    try:
        _ = import_module("modules.sd_samplers_kdiffusion")
        sampling = import_module("k_diffusion.sampling")

        BACKEND = "WebUI"
    except ImportError as _:
        pass
