from functools import partial

BACKEND = None

try:
    from comfy.ldm.modules.attention import optimized_attention
    from comfy.model_patcher import ModelPatcher
    from comfy.samplers import calc_cond_batch

    from .guidance_utils import (
        parse_unet_blocks,
        perturbed_attention,
        rescale_guidance,
        seg_attention_wrapper,
        snf_guidance,
        swg_pred_calc,
    )

    try:
        from comfy.model_patcher import set_model_options_patch_replace
    except ImportError:
        from .guidance_utils import set_model_options_patch_replace

    BACKEND = "ComfyUI"
except ImportError:
    from guidance_utils import (
        parse_unet_blocks,
        perturbed_attention,
        rescale_guidance,
        seg_attention_wrapper,
        set_model_options_patch_replace,
        snf_guidance,
        swg_pred_calc,
    )

    try:
        from ldm_patched.ldm.modules.attention import optimized_attention
        from ldm_patched.modules.model_patcher import ModelPatcher
        from ldm_patched.modules.samplers import calc_cond_uncond_batch

        BACKEND = "reForge"
    except ImportError:
        from backend.attention import attention_function as optimized_attention
        from backend.patcher.base import ModelPatcher
        from backend.sampling.sampling_function import calc_cond_uncond_batch

        BACKEND = "Forge"


class PerturbedAttention:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "model": ("MODEL",),
                "scale": ("FLOAT", {"default": 3.0, "min": 0.0, "max": 100.0, "step": 0.1, "round": 0.01}),
                "adaptive_scale": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 1.0, "step": 0.001, "round": 0.0001}),
                "unet_block": (["input", "middle", "output"], {"default": "middle"}),
                "unet_block_id": ("INT", {"default": 0}),
                "sigma_start": ("FLOAT", {"default": -1.0, "min": -1.0, "max": 10000.0, "step": 0.01, "round": False}),
                "sigma_end": ("FLOAT", {"default": -1.0, "min": -1.0, "max": 10000.0, "step": 0.01, "round": False}),
                "rescale": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 1.0, "step": 0.01}),
                "rescale_mode": (["full", "partial", "snf"], {"default": "full"}),
            },
            "optional": {
                "unet_block_list": ("STRING", {"default": ""}),
            },
        }

    RETURN_TYPES = ("MODEL",)
    FUNCTION = "patch"

    CATEGORY = "model_patches/unet"

    def patch(
        self,
        model: ModelPatcher,
        scale: float = 3.0,
        adaptive_scale: float = 0.0,
        unet_block: str = "middle",
        unet_block_id: int = 0,
        sigma_start: float = -1.0,
        sigma_end: float = -1.0,
        rescale: float = 0.0,
        rescale_mode: str = "full",
        unet_block_list: str = "",
    ):
        m = model.clone()

        sigma_start = float("inf") if sigma_start < 0 else sigma_start
        single_block = (unet_block, unet_block_id, None)
        blocks, block_names = (
            parse_unet_blocks(model, unet_block_list, "attn1") if unet_block_list else ([single_block], None)
        )

        def post_cfg_function(args):
            """CFG+PAG"""
            model = args["model"]
            cond_pred = args["cond_denoised"]
            uncond_pred = args["uncond_denoised"]
            cond = args["cond"]
            cfg_result = args["denoised"]
            sigma = args["sigma"]
            model_options = args["model_options"].copy()
            x = args["input"]

            signal_scale = scale
            if adaptive_scale > 0:
                t = 0
                if hasattr(model, "model_sampling"):
                    t = model.model_sampling.timestep(sigma)[0].item()
                else:
                    ts = model.predictor.timestep(sigma)
                    t = ts[0].item()
                signal_scale -= scale * (adaptive_scale**4) * (1000 - t)
                if signal_scale < 0:
                    signal_scale = 0

            if signal_scale == 0 or not (sigma_end < sigma[0] <= sigma_start):
                return cfg_result

            # Replace Self-attention with PAG
            for block in blocks:
                layer, number, index = block
                model_options = set_model_options_patch_replace(
                    model_options, perturbed_attention, "attn1", layer, number, index
                )

            if BACKEND == "ComfyUI":
                (pag_cond_pred,) = calc_cond_batch(model, [cond], x, sigma, model_options)
            if BACKEND in {"Forge", "reForge"}:
                (pag_cond_pred, _) = calc_cond_uncond_batch(model, cond, None, x, sigma, model_options)

            pag = (cond_pred - pag_cond_pred) * signal_scale

            if rescale_mode == "snf":
                if uncond_pred.any():
                    return uncond_pred + snf_guidance(cfg_result - uncond_pred, pag)
                return cfg_result + pag

            return cfg_result + rescale_guidance(pag, cond_pred, cfg_result, rescale, rescale_mode)

        m.set_model_sampler_post_cfg_function(post_cfg_function, rescale_mode == "snf")

        return (m,)


class SmoothedEnergyGuidanceAdvanced:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "model": ("MODEL",),
                "scale": ("FLOAT", {"default": 3.0, "min": 0.0, "max": 100.0, "step": 0.1, "round": 0.01}),
                "blur_sigma": ("FLOAT", {"default": -1.0, "min": -1.0, "max": 9999.0, "step": 0.01, "round": 0.001}),
                "unet_block": (["input", "middle", "output"], {"default": "middle"}),
                "unet_block_id": ("INT", {"default": 0}),
                "sigma_start": ("FLOAT", {"default": -1.0, "min": -1.0, "max": 10000.0, "step": 0.01, "round": False}),
                "sigma_end": ("FLOAT", {"default": -1.0, "min": -1.0, "max": 10000.0, "step": 0.01, "round": False}),
                "rescale": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 1.0, "step": 0.01}),
                "rescale_mode": (["full", "partial", "snf"], {"default": "full"}),
            },
            "optional": {
                "unet_block_list": ("STRING", {"default": ""}),
            },
        }

    RETURN_TYPES = ("MODEL",)
    FUNCTION = "patch"

    CATEGORY = "model_patches/unet"

    def patch(
        self,
        model: ModelPatcher,
        scale: float = 3.0,
        blur_sigma: float = -1.0,
        unet_block: str = "middle",
        unet_block_id: int = 0,
        sigma_start: float = -1.0,
        sigma_end: float = -1.0,
        rescale: float = 0.0,
        rescale_mode: str = "full",
        unet_block_list: str = "",
    ):
        m = model.clone()

        sigma_start = float("inf") if sigma_start < 0 else sigma_start
        single_block = (unet_block, unet_block_id, None)
        blocks, block_names = (
            parse_unet_blocks(model, unet_block_list, "attn1") if unet_block_list else ([single_block], None)
        )

        def post_cfg_function(args):
            """CFG+SEG"""
            model = args["model"]
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

            seg_attention = seg_attention_wrapper(optimized_attention, blur_sigma)

            # Replace Self-attention with SEG attention
            for block in blocks:
                layer, number, index = block
                model_options = set_model_options_patch_replace(
                    model_options, seg_attention, "attn1", layer, number, index
                )

            if BACKEND == "ComfyUI":
                (seg_cond_pred,) = calc_cond_batch(model, [cond], x, sigma, model_options)
            if BACKEND in {"Forge", "reForge"}:
                (seg_cond_pred, _) = calc_cond_uncond_batch(model, cond, None, x, sigma, model_options)

            seg = (cond_pred - seg_cond_pred) * signal_scale

            if rescale_mode == "snf":
                if uncond_pred.any():
                    return uncond_pred + snf_guidance(cfg_result - uncond_pred, seg)
                return cfg_result + seg

            return cfg_result + rescale_guidance(seg, cond_pred, cfg_result, rescale, rescale_mode)

        m.set_model_sampler_post_cfg_function(post_cfg_function, rescale_mode == "snf")

        return (m,)


class SlidingWindowGuidanceAdvanced:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "model": ("MODEL",),
                "scale": ("FLOAT", {"default": 5.0, "min": 0.0, "max": 100.0, "step": 0.1, "round": 0.01}),
                "tile_width": ("INT", {"default": 768, "min": 16, "max": 16384, "step": 8}),
                "tile_height": ("INT", {"default": 768, "min": 16, "max": 16384, "step": 8}),
                "tile_overlap": ("INT", {"default": 256, "min": 16, "max": 16384, "step": 8}),
                "sigma_start": ("FLOAT", {"default": -1.0, "min": -1.0, "max": 10000.0, "step": 0.01, "round": False}),
                "sigma_end": ("FLOAT", {"default": 5.42, "min": -1.0, "max": 10000.0, "step": 0.01, "round": False}),
            },
        }

    RETURN_TYPES = ("MODEL",)
    FUNCTION = "patch"

    CATEGORY = "model_patches/unet"

    def patch(
        self,
        model: ModelPatcher,
        scale: float = 5.0,
        tile_width: int = 768,
        tile_height: int = 768,
        tile_overlap: int = 256,
        sigma_start: float = -1.0,
        sigma_end: float = 5.42,
    ):
        m = model.clone()

        sigma_start = float("inf") if sigma_start < 0 else sigma_start
        tile_width, tile_height, tile_overlap = tile_width // 8, tile_height // 8, tile_overlap // 8

        def post_cfg_function(args):
            """CFG+SWG"""
            model = args["model"]
            cond_pred = args["cond_denoised"]
            cond = args["cond"]
            cfg_result = args["denoised"]
            sigma = args["sigma"]
            model_options = args["model_options"].copy()
            x = args["input"]

            signal_scale = scale

            if signal_scale == 0 or not (sigma_end < sigma[0] <= sigma_start):
                return cfg_result

            calc_func = None

            if BACKEND == "ComfyUI":
                calc_func = partial(
                    calc_cond_batch,
                    model=model,
                    conds=[cond],
                    timestep=sigma,
                    model_options=model_options,
                )
            if BACKEND in {"Forge", "reForge"}:
                calc_func = partial(
                    calc_cond_uncond_batch,
                    model=model,
                    cond=cond,
                    uncond=None,
                    timestep=sigma,
                    model_options=model_options,
                )

            swg_pred = swg_pred_calc(x, tile_width, tile_height, tile_overlap, calc_func)
            swg = (cond_pred - swg_pred) * signal_scale

            return cfg_result + swg

        m.set_model_sampler_post_cfg_function(post_cfg_function)

        return (m,)
