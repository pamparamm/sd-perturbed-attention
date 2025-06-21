import torch

import comfy.model_management
from comfy.model_base import BaseModel
from comfy.comfy_types.node_typing import IO, ComfyNodeABC, InputTypeDict
from comfy.ldm.modules.attention import BasicTransformerBlock, CrossAttention, optimized_attention
from comfy.model_patcher import ModelPatcher

from .pag_utils import parse_unet_blocks

COND = 0
UNCOND = 1


def nag_attn2_replace_wrapper(
    nag_scale: float,
    tau: float,
    alpha: float,
    sigma_start: float,
    sigma_end: float,
    k_neg: torch.Tensor,
    v_neg: torch.Tensor,
):
    # Modified Algorithm 1 from 2505.21179 'Normalized Attention Guidance: Universal Negative Guidance for Diffusion Models'
    def nag_attn2_replace(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, extra_options: dict):
        heads = extra_options["n_heads"]
        attn_precision = extra_options.get("attn_precision")
        sigma = extra_options["sigmas"]
        cond_or_uncond: list[int] = extra_options.get("cond_or_uncond")  # type: ignore

        # Perform batched CA
        z = optimized_attention(q, k, v, heads, attn_precision)

        if nag_scale == 0 or not (sigma_end < sigma[0] <= sigma_start) or COND not in cond_or_uncond:
            return z

        bs = q.shape[0] // len(cond_or_uncond) * cond_or_uncond.count(COND)

        k_neg_, v_neg_ = k_neg.repeat_interleave(bs, dim=0), v_neg.repeat_interleave(bs, dim=0)

        # Get conditional queries for NAG
        # Assume that cond_or_uncond has a layout [1, 1..., 0, 0...]
        q_chunked = q.chunk(len(cond_or_uncond))
        q_pos = torch.cat(q_chunked[cond_or_uncond.index(COND) :])

        # Apply NAG only to conditional parts of batched CA
        z_chunked = z.chunk(len(cond_or_uncond))
        z_pos = torch.cat(z_chunked[cond_or_uncond.index(COND) :])
        z_neg = optimized_attention(q_pos, k_neg_, v_neg_, heads, attn_precision)

        z_tilde = z_pos + nag_scale * (z_pos - z_neg)

        norm_pos = torch.norm(z_pos, p=1, dim=-1, keepdim=True)
        norm_tilde = torch.norm(z_tilde, p=1, dim=-1, keepdim=True)
        ratio = norm_tilde / norm_pos

        z_hat = torch.where(ratio > tau, tau, ratio) / ratio * z_tilde

        z_nag = alpha * z_hat + (1 - alpha) * z_pos

        # Prepend unconditional CA result to NAG result
        if UNCOND in cond_or_uncond:
            z_nag = torch.cat(z_chunked[cond_or_uncond.index(UNCOND) : cond_or_uncond.index(COND)] + (z_nag,))

        return z_nag

    return nag_attn2_replace


class NormalizedAttentionGuidance(ComfyNodeABC):
    @classmethod
    def INPUT_TYPES(cls) -> InputTypeDict:
        return {
            "required": {
                "model": (IO.MODEL, {}),
                "negative": (IO.CONDITIONING, {}),
                "scale": (IO.FLOAT, {"default": 2.0, "min": 0.0, "max": 100.0, "step": 0.1, "round": 0.01}),
                "tau": (IO.FLOAT, {"default": 2.5, "min": 0.0, "max": 100.0, "step": 0.1, "round": 0.01}),
                "alpha": (IO.FLOAT, {"default": 0.5, "min": 0.0, "max": 1.0, "step": 0.001}),
                "sigma_start": (IO.FLOAT, {"default": -1.0, "min": -1.0, "max": 10000.0, "step": 0.01, "round": False}),
                "sigma_end": (IO.FLOAT, {"default": -1.0, "min": -1.0, "max": 10000.0, "step": 0.01, "round": False}),
            },
            "optional": {
                "unet_block_list": (IO.STRING, {"default": ""}),
            },
        }

    RETURN_TYPES = (IO.MODEL,)

    FUNCTION = "patch"

    CATEGORY = "model_patches/unet"

    def patch(
        self,
        model: ModelPatcher,
        negative,
        scale=2.0,
        tau=2.5,
        alpha=0.5,
        sigma_start: float = -1.0,
        sigma_end: float = -1.0,
        unet_block_list: str = "str",
    ):
        m = model.clone()
        inner_model: BaseModel = m.model
        dtype = inner_model.get_dtype()
        if inner_model.manual_cast_dtype is not None:
            dtype = inner_model.manual_cast_dtype
        device_model = inner_model.device
        device_infer = comfy.model_management.get_torch_device()

        sigma_start = float("inf") if sigma_start < 0 else sigma_start

        negative_cond = negative[0][0].to(device_model, dtype=dtype)

        blocks = parse_unet_blocks(m, unet_block_list, attn="attn2") if unet_block_list else None

        for name, module in inner_model.diffusion_model.named_modules():
            # Apply NAG only to transformer blocks with cross-attention (attn2)
            if isinstance(module, BasicTransformerBlock) and getattr(module, "attn2", None):
                attn2: CrossAttention = module.attn2  # type: ignore
                parts: list[str] = name.split(".")
                block_name: str = parts[0].split("_")[0]
                block_id = int(parts[1])
                if block_name == "middle":
                    block_id = block_id - 1

                t_idx = None
                if "transformer_blocks" in parts:
                    t_pos = parts.index("transformer_blocks") + 1
                    t_idx = int(parts[t_pos])

                if not blocks or (block_name, block_id, t_idx) in blocks or (block_name, block_id, None) in blocks:
                    k_neg, v_neg = attn2.to_k(negative_cond), attn2.to_v(negative_cond)
                    nag_attn2_replace = nag_attn2_replace_wrapper(
                        scale,
                        tau,
                        alpha,
                        sigma_start,
                        sigma_end,
                        k_neg.to(device_infer, dtype=dtype),
                        v_neg.to(device_infer, dtype=dtype),
                    )
                    m.set_model_attn2_replace(nag_attn2_replace, block_name, block_id, t_idx)

        return (m,)
