import math
from typing import Callable
import torch
from torch import Tensor
import torch.nn.functional as F
from itertools import groupby


def parse_unet_blocks(model, unet_block_list: str):
    output: list[tuple[str, int, int | None]] = []

    # Get all Self-attention blocks
    input_blocks, middle_blocks, output_blocks = [], [], []
    for name, module in model.model.diffusion_model.named_modules():
        if module.__class__.__name__ == "CrossAttention" and name.endswith("attn1"):
            parts = name.split(".")
            block_name = parts[0]
            block_id = int(parts[1])
            if block_name.startswith("input"):
                input_blocks.append(block_id)
            elif block_name.startswith("middle"):
                middle_blocks.append(block_id - 1)
            elif block_name.startswith("output"):
                output_blocks.append(block_id)

    def group_blocks(blocks: list[int]):
        return [(i, len(list(gr))) for i, gr in groupby(blocks)]

    input_blocks, middle_blocks, output_blocks = group_blocks(input_blocks), group_blocks(middle_blocks), group_blocks(output_blocks)

    unet_blocks = [b.strip() for b in unet_block_list.split(",")]
    for block in unet_blocks:
        name, indices = block[0], block[1:].split(".")
        match name:
            case "d":
                layer, cur_blocks = "input", input_blocks
            case "m":
                layer, cur_blocks = "middle", middle_blocks
            case "u":
                layer, cur_blocks = "output", output_blocks
        if len(indices) >= 2:
            number, index = cur_blocks[int(indices[0])][0], int(indices[1])
            assert 0 <= index < cur_blocks[int(indices[0])][1]
        else:
            number, index = cur_blocks[int(indices[0])][0], None
        output.append((layer, number, index))

    return output


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


def perturbed_attention(q: Tensor, k: Tensor, v: Tensor, extra_options, mask=None):
    """Perturbed self-attention"""
    return v


# Modified 'Algorithm 2 Classifier-Free Guidance with Rescale' from Common Diffusion Noise Schedules and Sample Steps are Flawed (Lin et al.).
def rescale_guidance(guidance: torch.Tensor, cond_pred: torch.Tensor, cfg_result: torch.Tensor, rescale=0.0, rescale_mode="full"):
    if rescale == 0.0:
        return guidance

    match rescale_mode:
        case "full":
            guidance_result = cfg_result + guidance
        case _:
            guidance_result = cond_pred + guidance

    std_cond = torch.std(cond_pred, dim=(1, 2, 3), keepdim=True)
    std_guidance = torch.std(guidance_result, dim=(1, 2, 3), keepdim=True)

    factor = std_cond / std_guidance
    factor = rescale * factor + (1.0 - rescale)

    return guidance * factor


# Gaussian blur
def gaussian_blur_2d(img, kernel_size, sigma):
    height = img.shape[-1]
    kernel_size = min(kernel_size, height - (height % 2 - 1))
    ksize_half = (kernel_size - 1) * 0.5

    x = torch.linspace(-ksize_half, ksize_half, steps=kernel_size)

    pdf = torch.exp(-0.5 * (x / sigma).pow(2))

    x_kernel = pdf / pdf.sum()
    x_kernel = x_kernel.to(device=img.device, dtype=img.dtype)

    kernel2d = torch.mm(x_kernel[:, None], x_kernel[None, :])
    kernel2d = kernel2d.expand(img.shape[-3], 1, kernel2d.shape[0], kernel2d.shape[1])

    padding = [kernel_size // 2, kernel_size // 2, kernel_size // 2, kernel_size // 2]

    img = F.pad(img, padding, mode="reflect")
    img = F.conv2d(img, kernel2d, groups=img.shape[-3])

    return img


def seg_attention_wrapper(attention, blur_sigma=1.0):

    def seg_attention(q: Tensor, k: Tensor, v: Tensor, extra_options, mask=None):
        """Smoothed Energy Guidance self-attention"""
        heads = extra_options["n_heads"]
        bs, area, inner_dim = q.shape

        height_orig, width_orig = extra_options["original_shape"][2:4]
        aspect_ratio = width_orig / height_orig

        if aspect_ratio >= 1.0:
            height = round((area / aspect_ratio) ** 0.5)
            q = q.permute(0, 2, 1).reshape(bs, inner_dim, height, -1)
        else:
            width = round((area * aspect_ratio) ** 0.5)
            q = q.permute(0, 2, 1).reshape(bs, inner_dim, -1, width)

        if blur_sigma >= 0:
            kernel_size = math.ceil(6 * blur_sigma) + 1 - math.ceil(6 * blur_sigma) % 2
            q = gaussian_blur_2d(q, kernel_size, blur_sigma)
        else:
            q[:] = q.mean(dim=(-2, -1), keepdim=True)

        q = q.reshape(bs, inner_dim, -1).permute(0, 2, 1)

        return attention(q, k, v, heads=heads)

    return seg_attention


# Modified algorithm from 2411.10257 'The Unreasonable Effectiveness of Guidance for Diffusion Models' (Figure 6.)
def swg_pred_calc(x: Tensor, crop_count: int, crop_size: int, calc_func: Callable[..., tuple[Tensor]]):
    steps_per_dim = int(math.sqrt(crop_count))
    b, c, h, w = x.shape
    swg_pred = torch.zeros_like(x)
    overlap = torch.zeros_like(x)
    stride = (h - crop_size) // (steps_per_dim - 1)
    for i in range(steps_per_dim):
        for j in range(steps_per_dim):
            left, right = stride * i, stride * i + crop_size
            top, bottom = stride * j, stride * j + crop_size

            x_window = x[:, :, top:bottom, left:right]
            swg_pred_window = calc_func(x_in=x_window)[0]
            swg_pred[:, :, top:bottom, left:right] += swg_pred_window

            overlap_window = torch.ones_like(swg_pred_window)
            overlap[:, :, top:bottom, left:right] += overlap_window

    swg_pred = swg_pred / overlap
    return swg_pred


# Saliency-adaptive Noise Fusion based on High-fidelity Person-centric Subject-to-Image Synthesis (Wang et al.)
# https://github.com/CodeGoat24/Face-diffuser/blob/edff1a5178ac9984879d9f5e542c1d0f0059ca5f/facediffuser/pipeline.py#L535-L562
def snf_guidance(t_guidance: torch.Tensor, s_guidance: torch.Tensor):
    b, c, h, w = t_guidance.shape

    t_omega = gaussian_blur_2d(torch.abs(t_guidance), 3, 1)
    s_omega = gaussian_blur_2d(torch.abs(s_guidance), 3, 1)
    t_softmax = torch.softmax(t_omega.reshape(b * c, h * w), dim=1).reshape(b, c, h, w)
    s_softmax = torch.softmax(s_omega.reshape(b * c, h * w), dim=1).reshape(b, c, h, w)

    guidance_stacked = torch.stack([t_guidance, s_guidance], dim=0)
    ts_softmax = torch.stack([t_softmax, s_softmax], dim=0)

    argeps = torch.argmax(ts_softmax, dim=0, keepdim=True)

    snf = torch.gather(guidance_stacked, dim=0, index=argeps).squeeze(0)
    return snf
