import math
import kornia
from kornia.augmentation import PadTo
import torch
from kornia.geometry import build_laplacian_pyramid

from comfy.model_patcher import ModelPatcher
from comfy_api.latest import io

from .guidance_utils import project


# Implementation of 2506.19713 'Guidance in the Frequency Domain Enables High-Fidelity Sampling at Low CFG Scales'.
class FrequencyDecoupledGuidance(io.ComfyNode):
    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="FrequencyDecoupledGuidance",
            search_aliases=["fdg", "frequency decoupled guidance"],
            display_name="Frequency-Decoupled Guidance",
            category="model_patches/unet",
            inputs=[
                io.Model.Input("model"),
                io.Float.Input("strength_high", default=12.0, min=0.0, max=100.0, step=0.1, round=0.01),
            ],
            outputs=[
                io.Model.Output(),
            ],
        )

    @classmethod
    def execute(cls, **kwargs) -> io.NodeOutput:
        model: ModelPatcher = kwargs["model"]
        strength_high = kwargs["strength_high"]

        def fdg_cfg_function(args):
            cond_denoised: torch.Tensor = args["cond_denoised"]
            uncond_denoised: torch.Tensor = args["uncond_denoised"]
            cond_scale: float = args["cond_scale"]
            x_orig: torch.Tensor = args["input"]

            height, width = x_orig.shape[2:4]
            strength_low = cond_scale

            # Use padding if necessary (when latent dims are not divisible by 2)
            pad_op = cls.get_pad_op(height, width)

            if pad_op is not None:
                cond_denoised = pad_op(cond_denoised)
                uncond_denoised = pad_op(uncond_denoised)

            x_fdg = cls.laplacian_guidance(
                cond_denoised,
                uncond_denoised,
                strength_high,
                strength_low,
            )

            if pad_op is not None:
                x_fdg = pad_op.inverse(x_fdg)

            return x_orig - x_fdg

        m = model.clone()
        m.set_model_sampler_cfg_function(fdg_cfg_function)
        return io.NodeOutput(m)

    @classmethod
    def get_pad_op(cls, h: int, w: int) -> PadTo | None:
        h_ceil: int = 2 ** math.ceil(math.log2(h))
        w_ceil: int = 2 ** math.ceil(math.log2(w))
        if h_ceil == h and w_ceil == w:
            return None

        return PadTo((h_ceil, w_ceil))

    @classmethod
    def build_image_from_pyramid(cls, pyramid: list[torch.Tensor]):
        img = pyramid[-1]
        for i in range(len(pyramid) - 2, -1, -1):
            img = kornia.geometry.pyrup(img) + pyramid[i]
            del pyramid[i]
        return img

    @classmethod
    def laplacian_guidance(
        cls,
        cond: torch.Tensor,
        uncond: torch.Tensor,
        strength_high: float,
        strength_low: float,
    ) -> torch.Tensor:
        levels = 2

        cond_pyramid = build_laplacian_pyramid(cond, levels)
        uncond_pyramid = build_laplacian_pyramid(uncond, levels)

        guided_pyramid: list[torch.Tensor] = []

        parameters = zip(cond_pyramid, uncond_pyramid, [strength_high, strength_low])
        for _, (cond_i, uncond_i, scale) in enumerate(parameters):
            diff = cond_i - uncond_i
            diff_parallel, diff_orthogonal = project(diff, cond_i)
            diff = diff_parallel + diff_orthogonal
            guided_i = cond_i + (scale - 1.0) * diff
            guided_pyramid.append(guided_i)

        return cls.build_image_from_pyramid(guided_pyramid)


NODES = [FrequencyDecoupledGuidance]
