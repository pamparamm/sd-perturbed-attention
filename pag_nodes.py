from torch import Tensor

BACKEND = None

try:
    from comfy.model_patcher import ModelPatcher
    from comfy.samplers import calc_cond_batch
    try:
        from comfy.model_patcher import set_model_options_patch_replace
    except ImportError:
        from .pag_utils import set_model_options_patch_replace

    BACKEND = "ComfyUI"
except ImportError:
    from ldm_patched.modules.model_patcher import ModelPatcher
    from ldm_patched.modules.samplers import calc_cond_uncond_batch
    from pag_utils import set_model_options_patch_replace

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
            }
        }

    RETURN_TYPES = ("MODEL",)
    FUNCTION = "patch"

    CATEGORY = "advanced/model"

    def patch(self, model: ModelPatcher, scale: float = 3.0, adaptive_scale: float = 0.0, unet_block: str = "middle", unet_block_id: int = 0):
        m = model.clone()

        def perturbed_attention(q: Tensor, k: Tensor, v: Tensor, extra_options, mask=None):
            """Perturbed self-attention"""
            return v

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

            if signal_scale == 0:
                return cfg_result

            # Replace Self-attention with PAG
            model_options = set_model_options_patch_replace(model_options, perturbed_attention, "attn1", unet_block, unet_block_id)
            if BACKEND == "ComfyUI":
                (pag,) = calc_cond_batch(model, [cond], x, sigma, model_options)
            if BACKEND == "Forge":
                (pag, _) = calc_cond_uncond_batch(model, cond, None, x, sigma, model_options)

            return cfg_result + (cond_pred - pag) * signal_scale

        m.set_model_sampler_post_cfg_function(post_cfg_function, disable_cfg1_optimization=False)

        return (m,)
