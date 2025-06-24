import math
from itertools import groupby
from typing import Any, Callable, Literal

import torch
import torch.nn.functional as F


def parse_unet_blocks(model, unet_block_list: str, attn: Literal["attn1", "attn2"] | None):
    output: list[tuple[str, int, int | None]] = []
    names: list[str] = []

    # Get all Self-attention blocks
    input_blocks: list[tuple[int, str]] = []
    middle_blocks: list[tuple[int, str]] = []
    output_blocks: list[tuple[int, str]] = []
    for name, module in model.model.diffusion_model.named_modules():
        if module.__class__.__name__ == "BasicTransformerBlock" and (attn is None or hasattr(module, attn)):
            parts = name.split(".")
            unet_part = parts[0]
            block_id = int(parts[1])
            if unet_part.startswith("input"):
                input_blocks.append((block_id, name))
            elif unet_part.startswith("middle"):
                middle_blocks.append((block_id - 1, name))
            elif unet_part.startswith("output"):
                output_blocks.append((block_id, name))

    def group_blocks(blocks: list[tuple[int, str]]):
        grouped_blocks = [(i, list(gr)) for i, gr in groupby(blocks, lambda b: b[0])]
        return [(i, len(gr), list(idx[1] for idx in gr)) for i, gr in grouped_blocks]

    input_blocks_gr, middle_blocks_gr, output_blocks_gr = (
        group_blocks(input_blocks),
        group_blocks(middle_blocks),
        group_blocks(output_blocks),
    )

    user_inputs = [b.strip() for b in unet_block_list.split(",")]
    for user_input in user_inputs:
        unet_part_s, indices = user_input[0], user_input[1:].split(".")
        match unet_part_s:
            case "d":
                unet_part, unet_group = "input", input_blocks_gr
            case "m":
                unet_part, unet_group = "middle", middle_blocks_gr
            case "u":
                unet_part, unet_group = "output", output_blocks_gr
            case _:
                raise ValueError(f"Block {user_input}: Unknown block prefix {unet_part_s}")

        block_index_range = [int(b.strip()) for b in indices[0].split("-")]
        block_index_range_start = block_index_range[0]
        block_index_range_end = block_index_range[0] if len(block_index_range) != 2 else block_index_range[1]
        for block_index in range(block_index_range_start, block_index_range_end + 1):
            if block_index < 0 or block_index >= len(unet_group):
                raise ValueError(
                    f"Block {user_input}: Block index in out of range 0 <= {block_index} < {len(unet_group)}"
                )

            block_group = unet_group[block_index]
            block_index_real = block_group[0]

            if len(indices) == 1:
                output.append((unet_part, block_index_real, None))
                names.extend(block_group[2])
            else:
                transformer_index_range = [int(b.strip()) for b in indices[1].split("-")]
                transformer_index_range_start = transformer_index_range[0]
                transformer_index_range_end = (
                    transformer_index_range[0] if len(transformer_index_range) != 2 else transformer_index_range[1]
                )
                for transformer_index in range(transformer_index_range_start, transformer_index_range_end + 1):
                    if transformer_index is not None and (transformer_index < 0 or transformer_index >= block_group[1]):
                        raise ValueError(
                            f"Block {user_input}: Transformer index in out of range 0 <= {transformer_index} < {block_group[1]}"
                        )

                    output.append((unet_part, block_index_real, transformer_index))
                    names.append(block_group[2][transformer_index])

    return output, names


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


def set_model_options_value(model_options, key: str, value: Any):
    to = model_options["transformer_options"].copy()
    to[key] = value
    model_options["transformer_options"] = to
    return model_options


def perturbed_attention(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, extra_options, mask=None):
    """Perturbed self-attention"""
    return v


# Modified 'Algorithm 2 Classifier-Free Guidance with Rescale' from Common Diffusion Noise Schedules and Sample Steps are Flawed (Lin et al.).
def rescale_guidance(
    guidance: torch.Tensor, cond_pred: torch.Tensor, cfg_result: torch.Tensor, rescale=0.0, rescale_mode="full"
):
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
    def seg_attention(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, extra_options, mask=None):
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
def swg_pred_calc(
    x: torch.Tensor, tile_width: int, tile_height: int, tile_overlap: int, calc_func: Callable[..., tuple[torch.Tensor]]
):
    b, c, h, w = x.shape
    swg_pred = torch.zeros_like(x)
    overlap = torch.zeros_like(x)

    tiles_w = math.ceil(w / (tile_width - tile_overlap))
    tiles_h = math.ceil(h / (tile_height - tile_overlap))

    for w_i in range(tiles_w):
        for h_i in range(tiles_h):
            left, right = tile_width * w_i, tile_width * (w_i + 1) + tile_overlap
            top, bottom = tile_height * h_i, tile_height * (h_i + 1) + tile_overlap

            x_window = x[:, :, top:bottom, left:right]
            if x_window.shape[-1] == 0 or x_window.shape[-2] == 0:
                continue

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
