from typing_extensions import override

from comfy_api.latest import ComfyExtension, io

from . import fdg_nodes, mg_nodes, nag_nodes, pag_nodes, pladis_nodes, smc_nodes, tpg_nodes
from .compat.utils import v3_schema_stub


class SDPerturbedAttentionExtension(ComfyExtension):
    @override
    async def get_node_list(self) -> list[type[io.ComfyNode]]:
        # TODO convert more nodes to v3
        return [
            *fdg_nodes.NODES,
            *mg_nodes.NODES,
            *v3_schema_stub(nag_nodes),
            *v3_schema_stub(pag_nodes),
            *v3_schema_stub(pladis_nodes),
            *smc_nodes.NODES,
            *v3_schema_stub(tpg_nodes),
        ]


async def comfy_entrypoint():  # ComfyUI calls this to load your extension and its nodes.
    return SDPerturbedAttentionExtension()
