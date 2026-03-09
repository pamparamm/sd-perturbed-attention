from typing import Any

import torch

from comfy.model_patcher import ModelPatcher
from comfy_api.latest import io

from .guidance_utils import get_option_group


# Implementation of 2602.20360 'Momentum Guidance: Plug-and-Play Guidance for Flow Models'.
class MomentumGuidance(io.ComfyNode):
    MG_GROUP_KEY = "mg_params"
    M_T_KEY = "m_t"

    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="MomentumGuidance",
            search_aliases=["mg"],
            display_name="Momentum Guidance",
            category="model_patches/unet",
            inputs=[
                io.Model.Input("model"),
                io.Float.Input(
                    "momentum",
                    default=0.5,
                    min=0.0,
                    max=100.0,
                    step=0.01,
                    round=0.001,
                    tooltip="Momentum (alpha) hyperparameter. Controls the scale/strength of Momentum Guidance: setting value to 0.0 completely disables MG. 0.5 should be optimal for MG+CFG combination in most scenarios",
                ),
                io.Float.Input(
                    "ema",
                    default=0.6,
                    min=0.0,
                    max=1.0,
                    step=0.01,
                    round=0.001,
                    tooltip="EMA (beta) hyperparameter. Consider tuning this before tuning `momentum`",
                ),
            ],
            outputs=[
                io.Model.Output(),
            ],
        )

    @classmethod
    def execute(cls, **kwargs) -> io.NodeOutput:
        model: ModelPatcher = kwargs["model"]
        momentum: float = kwargs["momentum"]
        ema: float = kwargs["ema"]

        def mg_post_cfg_function(args: dict[str, Any]):
            """CFG+MG"""
            cfg_result = args["denoised"]
            model_options: dict[str, Any] = args["model_options"]

            mg_group = get_option_group(model_options, cls.MG_GROUP_KEY)
            m_t: torch.Tensor = mg_group.get(cls.M_T_KEY, cfg_result)

            x_mg = momentum * (cfg_result - m_t)

            m_t_next = (1 - ema) * cfg_result + ema * m_t
            mg_group[cls.M_T_KEY] = m_t_next

            return cfg_result + x_mg

        m = model.clone()
        m.set_model_sampler_post_cfg_function(mg_post_cfg_function)

        return io.NodeOutput(m)


NODES = [MomentumGuidance]
