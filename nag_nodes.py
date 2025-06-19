import torch

import comfy.model_management
from comfy.comfy_types.node_typing import IO, ComfyNodeABC, InputTypeDict
from comfy.ldm.modules.attention import BasicTransformerBlock, CrossAttention, optimized_attention
from comfy.model_patcher import ModelPatcher


def nag_attn2_replace_wrapper(nag_scale: float, tau: float, alpha: float, k_neg: torch.Tensor, v_neg: torch.Tensor):
    # Algorithm 1 from 2505.21179 'Normalized Attention Guidance: Universal Negative Guidance for Diffusion Models'
    def nag_attn2_replace(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, extra_options: dict):
        heads = extra_options["n_heads"]
        attn_precision = extra_options.get("attn_precision")
        cond_or_uncond = extra_options.get("cond_or_uncond")

        bs = q.shape[0] // len(cond_or_uncond)

        # TODO
        if len(cond_or_uncond) > 1:
            raise ValueError("NAG with CFG is not supported yet")

        k_pos, v_pos = k, v
        k_neg_, v_neg_ = k_neg.repeat_interleave(bs, dim=0), v_neg.repeat_interleave(bs, dim=0)

        z_pos = optimized_attention(q, k_pos, v_pos, heads, attn_precision)
        z_neg = optimized_attention(q, k_neg_, v_neg_, heads, attn_precision)

        z_tilde = z_pos + nag_scale * (z_pos - z_neg)

        norm_pos = torch.norm(z_pos, p=1, dim=-1, keepdim=True)
        norm_tilde = torch.norm(z_tilde, p=1, dim=-1, keepdim=True)
        ratio = norm_tilde / norm_pos

        z_hat = torch.where(ratio > tau, tau, ratio) / ratio * z_tilde

        z_nag = alpha * z_hat + (1 - alpha) * z_pos

        return z_nag

    return nag_attn2_replace


class NormalizedAttentionGuidance(ComfyNodeABC):
    @classmethod
    def INPUT_TYPES(cls) -> InputTypeDict:
        return {
            "required": {
                "model": (IO.MODEL, {}),
                "negative": (IO.CONDITIONING, {}),
                "nag_scale": (IO.FLOAT, {"default": 2.0, "min": 0.0, "max": 100.0, "step": 0.1, "round": 0.01}),
                "tau": (IO.FLOAT, {"default": 2.5, "min": 0.0, "max": 100.0, "step": 0.1, "round": 0.01}),
                "alpha": (IO.FLOAT, {"default": 0.5, "min": 0.0, "max": 1.0, "step": 0.001}),
            }
        }

    RETURN_TYPES = (IO.MODEL,)

    FUNCTION = "patch"

    CATEGORY = "model_patches/unet"
    EXPERIMENTAL = True

    def patch(
        self,
        model: ModelPatcher,
        negative,
        nag_scale=2.0,
        tau=2.5,
        alpha=0.5,
    ):
        m = model.clone()
        dtype = m.model.diffusion_model.dtype
        device = comfy.model_management.get_torch_device()

        negative_cond = negative[0][0].to(device, dtype=dtype)

        for name, module in m.model.diffusion_model.named_modules():
            # Apply NAG only to transformer blocks with cross-attention (attn2)
            if isinstance(module, BasicTransformerBlock) and getattr(module, "attn2", None):
                attn2: CrossAttention = module.attn2  # type: ignore
                k_neg, v_neg = attn2.to_k(negative_cond), attn2.to_v(negative_cond)
                parts: list[str] = name.split(".")
                block_name: str = parts[0].split("_")[0]
                block_id = int(parts[1])
                if block_name == "middle":
                    block_id = block_id - 1

                t_idx = None
                if "transformer_blocks" in parts:
                    t_pos = parts.index("transformer_blocks") + 1
                    t_idx = int(parts[t_pos])
                else:
                    pass

                m.set_model_attn2_replace(
                    nag_attn2_replace_wrapper(nag_scale, tau, alpha, k_neg, v_neg), block_name, block_id, t_idx
                )

        return (m,)
