# File: pag_trt_nodes.py

# These imports should be at the top of the file, with no indentation.
import comfy.model_patcher
from .guidance_utils import parse_unet_blocks, perturbed_attention, rescale_guidance


class TRTAttachPag:
    """
    This node modifies a model by replacing its self-attention layers with
    a Perturbed Attention Guidance (PAG) version. It also sets the PAG
    parameters, which are then compiled directly into the TensorRT engine.
    """
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "model": ("MODEL",),
                "unet_block": (["input", "middle", "output"], {"default": "middle"}),
                "unet_block_id": ("INT", {"default": 0}),
                # --- PAG PARAMETERS ADDED HERE ---
                "scale": ("FLOAT", {"default": 3.0, "min": 0.0, "max": 100.0, "step": 0.1, "round": 0.01}),
                "sigma_start": ("FLOAT", {"default": -1.0, "min": -1.0, "max": 10000.0, "step": 0.01, "round": False}),
                "sigma_end": ("FLOAT", {"default": -1.0, "min": -1.0, "max": 10000.0, "step": 0.01, "round": False}),
                "adaptive_scale": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 1.0, "step": 0.001, "round": 0.0001}),
                "rescale": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 1.0, "step": 0.01}),
                "rescale_mode": (["full", "partial"], {"default": "full"}),
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
        model: comfy.model_patcher.ModelPatcher,
        unet_block: str = "middle",
        unet_block_id: int = 0,
        # --- PAG PARAMETERS IN FUNCTION SIGNATURE ---
        scale: float = 3.0,
        sigma_start: float = -1.0,
        sigma_end: float = -1.0,
        adaptive_scale: float = 0.0,
        rescale: float = 0.0,
        rescale_mode: str = "full",
        unet_block_list: str = "",
    ):
        m = model.clone()

        # --- THIS IS THE NEW PART ---
        # We create a function that will inject our PAG parameters into the model's
        # context every time it's sampled. This makes the values available to the
        # perturbed_attention function.
        def pag_cfg_function(args):
            # The 'transformer_options' dict is the standard way to pass custom
            # data to a model's internal functions.
            transformer_options = args.get("transformer_options", {})
            
            # We add our PAG parameters to this dictionary.
            transformer_options["pag_scale"] = scale
            transformer_options["pag_sigma_start"] = sigma_start
            transformer_options["pag_sigma_end"] = sigma_end
            transformer_options["pag_adaptive_scale"] = adaptive_scale
            transformer_options["pag_rescale"] = rescale
            transformer_options["pag_rescale_mode"] = rescale_mode
            
            # We update the args dictionary with our modified options.
            args["transformer_options"] = transformer_options
            return args

        # We attach this function to the model. It will be called automatically
        # by the sampler before each model pass.
        m.set_model_sampler_cfg_function(pag_cfg_function)

        # --- This part remains the same ---
        single_block = (unet_block, unet_block_id, None)
        blocks, block_names = (
            parse_unet_blocks(model, unet_block_list, "attn1") if unet_block_list else ([single_block], None)
        )

        for block in blocks:
            layer, number, index = block
            m.set_model_attn1_replace(perturbed_attention, layer, number, index)

        return (m,)


class TRTPerturbedAttention:
    """
    This node combines a base model and a PAG model to apply the guidance
    during sampling. It takes two pre-generated TensorRT engines as input.
    """
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
        model_base: comfy.model_patcher.ModelPatcher,
        model_pag: comfy.model_patcher.ModelPatcher,
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

            # --- FINAL ATTEMPT ---
            # We check if the underlying model is available.
            if model_pag.model is None:
                print("Warning: PAG model's internal .model is None. Falling back to CFG result.")
                return cfg_result

            try:
                # The TRT model wrapper has a simple interface and only accepts the core arguments.
                # We call it directly without any extra options.
                pag_model_wrapper = model_pag.model
                (pag_cond_pred,) = pag_model_wrapper(x, sigma, [cond])
            except Exception as e:
                print(f"Error during PAG model forward pass: {e}")
                print("The attempt to call the TRT model wrapper with its core arguments failed.")
                print("This definitively confirms the architectural incompatibility of a dual-model PAG approach.")
                print("Falling back to CFG result.")
                return cfg_result

            pag = (cond_pred - pag_cond_pred) * signal_scale

            return cfg_result + rescale_guidance(pag, cond_pred, cfg_result, rescale, rescale_mode)

        m.set_model_sampler_post_cfg_function(post_cfg_function, disable_cfg1_optimization=True)

        return (m,)