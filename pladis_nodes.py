from comfy.comfy_types.node_typing import IO, ComfyNodeABC, InputTypeDict
from comfy.ldm.modules.attention import BasicTransformerBlock
from comfy.model_base import BaseModel
from comfy.model_patcher import ModelPatcher

from .guidance_utils import parse_unet_blocks
from .pladis_utils import SPARSE_FUNCTIONS, pladis_attention_wrapper


class Pladis(ComfyNodeABC):
    @classmethod
    def INPUT_TYPES(cls) -> InputTypeDict:
        return {
            "required": {
                "model": (IO.MODEL, {}),
                "scale": (IO.FLOAT, {"default": 2.0, "min": 0.0, "max": 100.0, "step": 0.1, "round": 0.01}),
                "sparse_func": (IO.COMBO, {"default": SPARSE_FUNCTIONS[0], "options": SPARSE_FUNCTIONS}),
            },
            "optional": {
                "unet_block_list": (
                    IO.STRING,
                    {
                        "default": "",
                        "tooltip": (
                            "Comma-separated blocks to which Pladis is being applied to. When the list is empty, PLADIS is being applied to all `u` and `d` blocks.\n"
                            "Read README from sd-perturbed-attention for more details."
                        ),
                    },
                ),
            },
        }

    RETURN_TYPES = (IO.MODEL,)
    FUNCTION = "patch"

    CATEGORY = "model_patches/unet"
    EXPERIMENTAL = True

    def patch(
        self,
        model: ModelPatcher,
        scale=2.0,
        sparse_func=SPARSE_FUNCTIONS[0],
        unet_block_list="",
    ):
        m = model.clone()
        inner_model: BaseModel = m.model
        pladis_attention = pladis_attention_wrapper(scale, sparse_func)

        blocks, block_names = parse_unet_blocks(m, unet_block_list, "attn2") if unet_block_list else (None, None)

        # Apply PLADIS only to transformer blocks with cross-attention (attn2)
        for name, module in (
            (n, m)
            for n, m in inner_model.diffusion_model.named_modules()
            if isinstance(m, BasicTransformerBlock) and getattr(m, "attn2", None)
        ):
            parts = name.split(".")
            block_name: str = parts[0].split("_")[0]
            block_id = int(parts[1])
            if block_name == "middle":
                block_id = block_id - 1
                if not blocks:
                    continue

            t_idx = None
            if "transformer_blocks" in parts:
                t_pos = parts.index("transformer_blocks") + 1
                t_idx = int(parts[t_pos])

            if not blocks or (block_name, block_id, t_idx) in blocks or (block_name, block_id, None) in blocks:
                m.set_model_attn2_replace(pladis_attention, block_name, block_id, t_idx)

        return (m,)


NODE_CLASS_MAPPINGS = {
    "PLADIS": Pladis,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "PLADIS": "PLADIS",
}
