from .pag_nodes import PerturbedAttention, TRTAttachPag, TRTPerturbedAttention

NODE_CLASS_MAPPINGS = {
    "PerturbedAttention": PerturbedAttention,
    "TRTAttachPag": TRTAttachPag,
    "TRTPerturbedAttention": TRTPerturbedAttention,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "PerturbedAttention": "Perturbed-Attention Guidance (Advanced)",
    "TRTAttachPag": "TensorRT Attach PAG",
    "TRTPerturbedAttention": "TensorRT Perturbed-Attention Guidance",
}
