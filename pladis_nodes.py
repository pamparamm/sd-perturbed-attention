from comfy.model_patcher import ModelPatcher
from .pladis_utils import ENTMAX15_FUNC, SPARSEMAX_FUNC, pladis_attention_wrapper


class Pladis:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "model": ("MODEL",),
                "scale": ("FLOAT", {"default": 2.0, "min": 0.0, "max": 100.0, "step": 0.1, "round": 0.01}),
                "sparse_func": (s.SPARSE_FUNCTIONS,),
            }
        }

    RETURN_TYPES = ("MODEL",)
    FUNCTION = "patch"

    CATEGORY = "model_patches/unet"

    SPARSE_FUNCTIONS = [ENTMAX15_FUNC, SPARSEMAX_FUNC]

    def patch(self, model: ModelPatcher, scale: float, sparse_func: str):
        m = model.clone()
        pladis_attention = pladis_attention_wrapper(scale, sparse_func)

        for name, module in m.model.diffusion_model.named_modules():
            # Apply PLADIS only to cross-attention layers (attn2)
            if module.__class__.__name__ == "CrossAttention" and name.endswith("attn2"):
                parts = name.split(".")
                block_name: str = parts[0].split("_")[0]
                block_id = int(parts[1])
                if block_name == "middle":
                    block_id = block_id - 1

                m.set_model_attn2_replace(pladis_attention, block_name, block_id)

        return (m,)
