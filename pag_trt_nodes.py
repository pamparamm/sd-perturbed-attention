from comfy.model_patcher import ModelPatcher
from comfy.samplers import calc_cond_batch

from .guidance_utils import parse_unet_blocks, perturbed_attention, rescale_guidance


class TRTAttachPag:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "model": ("MODEL",),
                "unet_block": (["input", "middle", "output"], {"default": "middle"}),
                "unet_block_id": ("INT", {"default": 0}),
            },
            "optional": {
                "unet_block_list": ("STRING", {"default": ""}),
            },
        }

    RETURN_TYPES = ("MODEL",)
    FUNCTION = "attach"

    CATEGORY = "TensorRT"

    def attach(
        self,
        model: ModelPatcher,
        unet_block: str = "middle",
        unet_block_id: int = 0,
        unet_block_list: str = "",
    ):
        m = model.clone()

        single_block = (unet_block, unet_block_id, None)
        blocks, block_names = (
            parse_unet_blocks(model, unet_block_list, "attn1") if unet_block_list else ([single_block], None)
        )

        # Replace Self-attention with PAG
        for block in blocks:
            layer, number, index = block
            m.set_model_attn1_replace(perturbed_attention, layer, number, index)

        return (m,)


class TRTPerturbedAttention:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "model_base": ("MODEL",),
                "model_pag": ("MODEL",),
                "scale": ("FLOAT", {"default": 3.0, "min": 0.0, "max": 100.0, "step": 0.1, "round": 0.01}),
                "adaptive_scale": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 1.0, "step": 0.001, "round": 0.0001}),
                "sigma_start": ("FLOAT", {"default": -1.0, "min": -1.0, "max": 10000.0, "step": 0.01, "round": False}),
                "sigma_end": ("FLOAT", {"default": -1.0, "min": -1.0, "max": 10000.0, "step": 0.01, "round": False}),
                "rescale": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 1.0, "step": 0.01}),
                "rescale_mode": (["full", "partial"], {"default": "full"}),
            },
        }

    RETURN_TYPES = ("MODEL",)
    FUNCTION = "patch"

    CATEGORY = "TensorRT"

    def patch(
        self,
        model_base: ModelPatcher,
        model_pag: ModelPatcher,
        scale: float = 3.0,
        adaptive_scale: float = 0.0,
        sigma_start: float = -1.0,
        sigma_end: float = -1.0,
        rescale: float = 0.0,
        rescale_mode: str = "full",
    ):
        m = model_base.clone()

        sigma_start = float("inf") if sigma_start < 0 else sigma_start

        def post_cfg_function(args):
            """CFG+PAG"""
            model = args["model"]
            cond_pred = args["cond_denoised"]
            cond = args["cond"]
            cfg_result = args["denoised"]
            sigma = args["sigma"]
            x = args["input"]

            signal_scale = scale
            if adaptive_scale > 0:
                t = model.model_sampling.timestep(sigma)[0].item()
                signal_scale -= scale * (adaptive_scale**4) * (1000 - t)
                if signal_scale < 0:
                    signal_scale = 0

            if signal_scale == 0 or not (sigma_end < sigma[0] <= sigma_start):
                return cfg_result

            (pag_cond_pred,) = calc_cond_batch(model_pag.model, [cond], x, sigma, model_pag.model_options)

            pag = (cond_pred - pag_cond_pred) * signal_scale

            return cfg_result + rescale_guidance(pag, cond_pred, cfg_result, rescale, rescale_mode)

        m.set_model_sampler_post_cfg_function(post_cfg_function)

        return (m,)
