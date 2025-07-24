from contextlib import suppress
from typing import Callable

import torch

BACKEND = None

try:
    from ldm_patched.ldm.modules.attention import optimized_attention
    from ldm_patched.modules.model_patcher import ModelPatcher

    BACKEND = "reForge"
except ImportError:
    from backend.attention import attention_function as optimized_attention
    from backend.patcher.base import ModelPatcher

    BACKEND = "Forge"

# Try relative import first
try:
    from .guidance_utils import parse_unet_blocks, set_model_options_patch_replace
except ImportError:
    # Fallback
    from guidance_utils import parse_unet_blocks, set_model_options_patch_replace

COND = 0
UNCOND = 1

def nag_attn2_replace_wrapper(
    nag_scale: float,
    tau: float,
    alpha: float,
    sigma_start: float,
    sigma_end: float,
    negative_cond,  # Can be tensor or dict for SDXL
    attn2_module,
    prev_attn2_replace: Callable | None = None,
):
    # Modified Algorithm 1 from https://ar5iv.labs.arxiv.org/html/2505.21179 'Normalized Attention Guidance'
    def nag_attn2_replace(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, extra_options: dict):
        heads = extra_options["n_heads"]
        attn_precision = extra_options.get("attn_precision")
        sigma = extra_options["sigmas"]
        cond_or_uncond: list[int] = extra_options.get("cond_or_uncond")  # type: ignore

        # Batched CA
        z = (
            optimized_attention(q, k, v, heads, attn_precision)
            if prev_attn2_replace is None
            else prev_attn2_replace(q, k, v, extra_options)
        )

        if nag_scale == 0 or not (sigma_end < sigma[0] <= sigma_start) or COND not in cond_or_uncond:
            return z

        device = q.device
        dtype = q.dtype

        if isinstance(negative_cond, dict):
            # SDXL format: {'crossattn': tensor, 'vector': tensor}
            negative_cond_tensor = negative_cond.get('crossattn', negative_cond.get('c_crossattn'))
            if negative_cond_tensor is None:
                raise ValueError("Could not find crossattn tensor in SDXL conditioning dict")
            negative_cond_tensor = negative_cond_tensor.to(device, dtype=dtype)
        else:
            # Regular SD format: just a tensor
            negative_cond_tensor = negative_cond.to(device, dtype=dtype)

        k_neg = attn2_module.to_k(negative_cond_tensor)
        v_neg = attn2_module.to_v(negative_cond_tensor)

        bs = q.shape[0] // len(cond_or_uncond) * cond_or_uncond.count(COND)
        k_neg_, v_neg_ = k_neg.repeat_interleave(bs, dim=0), v_neg.repeat_interleave(bs, dim=0)

        if len(cond_or_uncond) == 1:
            # CFG=1 case: only conditional, no chunking needed
            q_pos = q
            z_pos = z
        else:
            # CFG>1 case: need to extract conditional part
            q_chunked = q.chunk(len(cond_or_uncond))
            q_pos = torch.cat(q_chunked[cond_or_uncond.index(COND) :])
            z_chunked = z.chunk(len(cond_or_uncond))
            z_pos = torch.cat(z_chunked[cond_or_uncond.index(COND) :])

        z_neg = optimized_attention(q_pos, k_neg_, v_neg_, heads, attn_precision)

        z_tilde = z_pos + nag_scale * (z_pos - z_neg)

        norm_pos = torch.norm(z_pos, p=1, dim=-1, keepdim=True)
        norm_tilde = torch.norm(z_tilde, p=1, dim=-1, keepdim=True)
        ratio = norm_tilde / norm_pos

        # Tau threshold
        z_hat = torch.where(ratio > tau, tau, ratio) / ratio * z_tilde

        z_nag = alpha * z_hat + (1 - alpha) * z_pos

        # Handle return based on whether we have unconditional part
        if len(cond_or_uncond) == 1:
            # CFG=1 case: return NAG result directly
            return z_nag
        else:
            # CFG>1 case: prepend unconditional CA result to NAG result
            if UNCOND in cond_or_uncond:
                z_chunked = z.chunk(len(cond_or_uncond))
                z_nag = torch.cat(z_chunked[cond_or_uncond.index(UNCOND) : cond_or_uncond.index(COND)] + (z_nag,))
            return z_nag

    return nag_attn2_replace


class NormalizedAttentionGuidance:
    def __init__(self):
        pass

    def patch(
        self,
        model: ModelPatcher,
        negative_cond,
        scale: float = 2.0,
        tau: float = 2.5,
        alpha: float = 0.5,
        sigma_start: float = -1.0,
        sigma_end: float = -1.0,
        unet_block_list: str = "",
    ):
        m = model.clone()
        inner_model: BaseModel = m.model

        sigma_start = float("inf") if sigma_start < 0 else sigma_start

        blocks, block_names = parse_unet_blocks(m, unet_block_list, "attn2") if unet_block_list else (None, None)

        if BACKEND == "reForge":
            from ldm_patched.ldm.modules.attention import BasicTransformerBlock, CrossAttention
        else:
            from backend.nn.unet import BasicTransformerBlock, CrossAttention

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
                    # Compatibility with other attn2 replaces (such as IPAdapter)
                    prev_attn2_replace = None
                    with suppress(KeyError):
                        block = (block_name, block_id, t_idx)
                        block_full = (block_name, block_id)
                        attn2_patches = m.model_options["transformer_options"]["patches_replace"]["attn2"]
                        if block_full in attn2_patches:
                            prev_attn2_replace = attn2_patches[block_full]
                        elif block in attn2_patches:
                            prev_attn2_replace = attn2_patches[block]

                    # Pass negative conditioning and attn2 module for on-demand computation
                    nag_attn2_replace = nag_attn2_replace_wrapper(
                        scale,
                        tau,
                        alpha,
                        sigma_start,
                        sigma_end,
                        negative_cond,
                        attn2,
                        prev_attn2_replace,
                    )
                    m.set_model_attn2_replace(nag_attn2_replace, block_name, block_id, t_idx)
        return m
