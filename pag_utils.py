import torch


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


# Modified 'Algorithm 2 Classifier-Free Guidance with Rescale' from Common Diffusion Noise Schedules and Sample Steps are Flawed (Lin et al.).
def rescale_pag(pag: torch.Tensor, cond_pred: torch.Tensor, rescale=0.0):
    if rescale == 0.0:
        return pag

    std_cond = torch.std(cond_pred, dim=(1, 2, 3), keepdim=True)
    std_pag = torch.std(cond_pred + pag, dim=(1, 2, 3), keepdim=True)

    factor = std_cond / std_pag
    factor = rescale * factor + (1.0 - rescale)

    return pag * factor
