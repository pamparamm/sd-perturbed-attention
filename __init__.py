from .nag_nodes import NormalizedAttentionGuidance
from .pag_nodes import PerturbedAttention, SlidingWindowGuidanceAdvanced, SmoothedEnergyGuidanceAdvanced
from .pag_trt_nodes import TRTAttachPag, TRTPerturbedAttention
from .pladis_nodes import Pladis

NODE_CLASS_MAPPINGS = {
    "PerturbedAttention": PerturbedAttention,
    "SmoothedEnergyGuidanceAdvanced": SmoothedEnergyGuidanceAdvanced,
    "SlidingWindowGuidanceAdvanced": SlidingWindowGuidanceAdvanced,
    "Pladis": Pladis,
    "NormalizedAttentionGuidance": NormalizedAttentionGuidance,
    "TRTAttachPag": TRTAttachPag,
    "TRTPerturbedAttention": TRTPerturbedAttention,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "PerturbedAttention": "Perturbed-Attention Guidance (Advanced)",
    "SmoothedEnergyGuidanceAdvanced": "Smoothed Energy Guidance (Advanced)",
    "SlidingWindowGuidanceAdvanced": "Sliding Window Guidance (Advanced)",
    "Pladis": "PLADIS (Experimental)",
    "NormalizedAttentionGuidance": "Normalized Attention Guidance",
    "TRTAttachPag": "TensorRT Attach PAG",
    "TRTPerturbedAttention": "TensorRT Perturbed-Attention Guidance",
}
