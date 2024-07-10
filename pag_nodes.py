from itertools import groupby
from torch import Tensor

BACKEND = None

try:
    from comfy.model_patcher import ModelPatcher
    from comfy.samplers import calc_cond_batch
    from .pag_utils import rescale_pag

    try:
        from comfy.model_patcher import set_model_options_patch_replace
    except ImportError:
        from .pag_utils import set_model_options_patch_replace

    BACKEND = "ComfyUI"
except ImportError:
    from ldm_patched.modules.model_patcher import ModelPatcher
    from ldm_patched.modules.samplers import calc_cond_uncond_batch
    from pag_utils import set_model_options_patch_replace, rescale_pag

    BACKEND = "Forge"


def perturbed_attention(q: Tensor, k: Tensor, v: Tensor, extra_options, mask=None):
    """Perturbed self-attention"""
    return v


def parse_unet_blocks(model: ModelPatcher, unet_block_list: str):
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
                "rescale_mode": (["full", "partial"], {"default": "full"}),
            },
            "optional": {
                "unet_block_list": ("STRING", {"default": ""}),
            },
        }

    RETURN_TYPES = ("MODEL",)
    FUNCTION = "patch"

    CATEGORY = "advanced/model"

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
        if unet_block_list:
            blocks = parse_unet_blocks(model, unet_block_list)
        else:
            blocks = [(unet_block, unet_block_id, None)]

        def post_cfg_function(args):
            """CFG+PAG"""
            model = args["model"]
            cond_pred = args["cond_denoised"]
            cond = args["cond"]
            cfg_result = args["denoised"]
            sigma = args["sigma"]
            model_options = args["model_options"].copy()
            x = args["input"]

            signal_scale = scale
            if adaptive_scale > 0:
                t = model.model_sampling.timestep(sigma)[0].item()
                signal_scale -= scale * (adaptive_scale**4) * (1000 - t)
                if signal_scale < 0:
                    signal_scale = 0

            if signal_scale == 0 or not (sigma_end < sigma[0] <= sigma_start):
                return cfg_result

            # Replace Self-attention with PAG
            for block in blocks:
                layer, number, index = block
                model_options = set_model_options_patch_replace(model_options, perturbed_attention, "attn1", layer, number, index)

            if BACKEND == "ComfyUI":
                (pag_cond_pred,) = calc_cond_batch(model, [cond], x, sigma, model_options)
            if BACKEND == "Forge":
                (pag_cond_pred, _) = calc_cond_uncond_batch(model, cond, None, x, sigma, model_options)

            pag = (cond_pred - pag_cond_pred) * signal_scale

            return cfg_result + rescale_pag(pag, cond_pred, cfg_result, rescale, rescale_mode)

        m.set_model_sampler_post_cfg_function(post_cfg_function)

        return (m,)


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

        if unet_block_list:
            blocks = parse_unet_blocks(model, unet_block_list)
        else:
            blocks = [(unet_block, unet_block_id, None)]

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

            return cfg_result + rescale_pag(pag, cond_pred, cfg_result, rescale, rescale_mode)

        m.set_model_sampler_post_cfg_function(post_cfg_function)

        return (m,)
