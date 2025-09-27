from . import nag_nodes, tpg_nodes, pladis_nodes
from .pag_nodes import PerturbedAttention, SlidingWindowGuidanceAdvanced, SmoothedEnergyGuidanceAdvanced
from .pag_trt_nodes import TRTAttachPag, TRTPerturbedAttention

NODE_CLASS_MAPPINGS = {
    "PerturbedAttention": PerturbedAttention,
    "SmoothedEnergyGuidanceAdvanced": SmoothedEnergyGuidanceAdvanced,
    "SlidingWindowGuidanceAdvanced": SlidingWindowGuidanceAdvanced,
    "TRTAttachPag": TRTAttachPag,
    "TRTPerturbedAttention": TRTPerturbedAttention,
    **nag_nodes.NODE_CLASS_MAPPINGS,
    **tpg_nodes.NODE_CLASS_MAPPINGS,
    **pladis_nodes.NODE_CLASS_MAPPINGS,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "PerturbedAttention": "Perturbed-Attention Guidance (Advanced)",
    "SmoothedEnergyGuidanceAdvanced": "Smoothed Energy Guidance (Advanced)",
    "SlidingWindowGuidanceAdvanced": "Sliding Window Guidance (Advanced)",
    "TRTAttachPag": "TensorRT Attach PAG",
    "TRTPerturbedAttention": "TensorRT Perturbed-Attention Guidance",
    **nag_nodes.NODE_DISPLAY_NAME_MAPPINGS,
    **tpg_nodes.NODE_DISPLAY_NAME_MAPPINGS,
    **pladis_nodes.NODE_DISPLAY_NAME_MAPPINGS,
}
