from . import nag_nodes, tpg_nodes
from .pag_nodes import PerturbedAttention, SlidingWindowGuidanceAdvanced, SmoothedEnergyGuidanceAdvanced
from .pag_trt_nodes import TRTAttachPag, TRTPerturbedAttention
from .pladis_nodes import Pladis

NODE_CLASS_MAPPINGS = {
    "PerturbedAttention": PerturbedAttention,
    "SmoothedEnergyGuidanceAdvanced": SmoothedEnergyGuidanceAdvanced,
    "SlidingWindowGuidanceAdvanced": SlidingWindowGuidanceAdvanced,
    "Pladis": Pladis,
    "TRTAttachPag": TRTAttachPag,
    "TRTPerturbedAttention": TRTPerturbedAttention,
    **nag_nodes.NODE_CLASS_MAPPINGS,
    **tpg_nodes.NODE_CLASS_MAPPINGS,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "PerturbedAttention": "Perturbed-Attention Guidance (Advanced)",
    "SmoothedEnergyGuidanceAdvanced": "Smoothed Energy Guidance (Advanced)",
    "SlidingWindowGuidanceAdvanced": "Sliding Window Guidance (Advanced)",
    "Pladis": "PLADIS (Experimental)",
    "TRTAttachPag": "TensorRT Attach PAG",
    "TRTPerturbedAttention": "TensorRT Perturbed-Attention Guidance",
    **nag_nodes.NODE_DISPLAY_NAME_MAPPINGS,
    **tpg_nodes.NODE_DISPLAY_NAME_MAPPINGS,
}
