from typing import Any

import torch
from torch import nn

from comfy.comfy_types.node_typing import IO, ComfyNodeABC, InputTypeDict
from comfy.ldm.modules.attention import BasicTransformerBlock
from comfy.model_base import BaseModel
from comfy.model_patcher import ModelPatcher
from comfy.samplers import calc_cond_batch

from .guidance_utils import parse_unet_blocks, rescale_guidance, set_model_options_value, snf_guidance

TPG_OPTION = "tpg"


# Implementation of 2506.10036 'Token Perturbation Guidance for Diffusion Models'
class TPGTransformerWrapper(nn.Module):
    def __init__(self, transformer_block: BasicTransformerBlock) -> None:
        super().__init__()
        self.wrapped_block = transformer_block

    def shuffle_tokens(self, x: torch.Tensor):
        # ComfyUI's torch.manual_seed generator should produce the same results here.
        permutation = torch.randperm(x.shape[1], device=x.device)
        return x[:, permutation]

    def forward(self, x: torch.Tensor, context: torch.Tensor | None = None, transformer_options: dict[str, Any] = {}):
        is_tpg = transformer_options.get(TPG_OPTION, False)
        x_ = self.shuffle_tokens(x) if is_tpg else x
        return self.wrapped_block(x_, context=context, transformer_options=transformer_options)


class TokenPerturbationGuidance(ComfyNodeABC):
    @classmethod
    def INPUT_TYPES(cls) -> InputTypeDict:
        return {
            "required": {
                "model": (IO.MODEL, {}),
                "scale": (IO.FLOAT, {"default": 3.0, "min": 0.0, "max": 100.0, "step": 0.1, "round": 0.01}),
                "sigma_start": (IO.FLOAT, {"default": -1.0, "min": -1.0, "max": 10000.0, "step": 0.01, "round": False}),
                "sigma_end": (IO.FLOAT, {"default": -1.0, "min": -1.0, "max": 10000.0, "step": 0.01, "round": False}),
                "rescale": (IO.FLOAT, {"default": 0.0, "min": 0.0, "max": 1.0, "step": 0.01}),
                "rescale_mode": (IO.COMBO, {"options": ["full", "partial", "snf"], "default": "full"}),
            },
            "optional": {
                "unet_block_list": (IO.STRING, {"default": "d2,d3", "tooltip": "Blocks to which TPG is applied. "}),
            },
        }

    RETURN_TYPES = (IO.MODEL,)
    FUNCTION = "patch"

    CATEGORY = "model_patches/unet"

    def patch(
        self,
        model: ModelPatcher,
        scale: float = 3.0,
        sigma_start: float = -1.0,
        sigma_end: float = -1.0,
        rescale: float = 0.0,
        rescale_mode: str = "full",
        unet_block_list: str = "",
    ):
        m = model.clone()
        inner_model: BaseModel = m.model

        sigma_start = float("inf") if sigma_start < 0 else sigma_start

        blocks, block_names = parse_unet_blocks(model, unet_block_list, None) if unet_block_list else (None, None)

        # Patch transformer blocks with TPG wrapper
        for name, module in inner_model.diffusion_model.named_modules():
            if (
                isinstance(module, BasicTransformerBlock)
                and not "wrapped_block" in name
                and (block_names is None or name in block_names)
            ):
                # Potential memory leak?
                wrapper = TPGTransformerWrapper(module)
                m.add_object_patch(f"diffusion_model.{name}", wrapper)

        def post_cfg_function(args):
            """CFG+TPG"""
            model: BaseModel = args["model"]
            cond_pred = args["cond_denoised"]
            uncond_pred = args["uncond_denoised"]
            cond = args["cond"]
            cfg_result = args["denoised"]
            sigma = args["sigma"]
            model_options = args["model_options"].copy()
            x = args["input"]

            signal_scale = scale

            if signal_scale == 0 or not (sigma_end < sigma[0] <= sigma_start):
                return cfg_result

            # Enable TPG in patched transformer blocks
            for name, module in model.diffusion_model.named_modules():
                if isinstance(module, TPGTransformerWrapper):
                    set_model_options_value(model_options, TPG_OPTION, True)

            (tpg_cond_pred,) = calc_cond_batch(model, [cond], x, sigma, model_options)

            tpg = (cond_pred - tpg_cond_pred) * signal_scale

            if rescale_mode == "snf":
                if uncond_pred.any():
                    return uncond_pred + snf_guidance(cfg_result - uncond_pred, tpg)
                return cfg_result + tpg

            return cfg_result + rescale_guidance(tpg, cond_pred, cfg_result, rescale, rescale_mode)

        m.set_model_sampler_post_cfg_function(post_cfg_function, rescale_mode == "snf")

        return (m,)


NODE_CLASS_MAPPINGS = {
    "TokenPerturbationGuidance": TokenPerturbationGuidance,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "TokenPerturbationGuidance": "Token Perturbation Guidance",
}
