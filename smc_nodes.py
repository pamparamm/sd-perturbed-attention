from typing import Any

import torch
from torch.linalg import vector_norm

from comfy.model_patcher import ModelPatcher
from comfy_api.latest import io

from .guidance_utils import get_option_group


# Implementation of 2603.03281 'CFG-Ctrl: Control-Based Classifier-Free Diffusion Guidance'.
class SlidingModeControlCFG(io.ComfyNode):
    SMC_GROUP_KEY = "smc_params"
    E_T_PREV_KEY = "e_t_prev"

    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="SlidingModeControlCFG",
            search_aliases=["smc", "smc cfg", "sliding mode control cfg"],
            display_name="SMC-CFG",
            category="model_patches/unet",
            is_experimental=True,
            inputs=[
                io.Model.Input("model"),
                io.Float.Input(
                    "lambda",
                    default=6.0,
                    min=0.0,
                    max=100.0,
                    step=0.1,
                    round=0.01,
                    tooltip="Shape hyperparameter of the sliding mode surface. Too high/low values may lead to guidance instabilities. Values from range [2.0, 8.0] were used in the original paper",
                ),
                io.Float.Input(
                    "k",
                    default=0.1,
                    min=0.0,
                    max=2.0,
                    step=0.01,
                    round=0.001,
                    tooltip="Force hyperparameter, controls the force towards the sliding mode surface. Low values may weaken text-image alignment but increase overall realism/aesthetic. Values from range [0.01, 0.8] were used in the original paper",
                ),
            ],
            outputs=[
                io.Model.Output(),
            ],
        )

    @classmethod
    def execute(cls, **kwargs) -> io.NodeOutput:
        model: ModelPatcher = kwargs["model"]
        l: float = kwargs["lambda"]
        k: float = kwargs["k"]

        def smc_cfg_function(args):
            cond_denoised: torch.Tensor = args["cond_denoised"]
            uncond_denoised: torch.Tensor = args["uncond_denoised"]
            cond_scale: float = args["cond_scale"]
            x_orig: torch.Tensor = args["input"]
            model_options: dict[str, Any] = args["model_options"]

            smc_group = get_option_group(model_options, cls.SMC_GROUP_KEY)

            e_t = cond_denoised - uncond_denoised
            e_t_prev: torch.Tensor = smc_group.get(cls.E_T_PREV_KEY, e_t)

            s_t = (e_t - e_t_prev) + l * e_t_prev

            # I'm using `unit_2(s_t)` instead of `sign(s_t)` here, since Table 4. from the original paper states that sign(s_t)==unit_2(s_t), and for some reason the former doesn't work at all (?_?).
            s_t_unit = s_t / vector_norm(s_t, dim=(1, 2, 3), keepdim=True)
            e_t_delta = -k * s_t_unit

            e_t_upd = e_t + e_t_delta

            smc_group[cls.E_T_PREV_KEY] = e_t_upd

            x_smc = uncond_denoised + cond_scale * e_t_upd

            return x_orig - x_smc

        m = model.clone()
        m.set_model_sampler_cfg_function(smc_cfg_function)

        return io.NodeOutput(m)


NODES = [SlidingModeControlCFG]
