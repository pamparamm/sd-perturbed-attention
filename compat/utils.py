from dataclasses import dataclass
from types import ModuleType

from comfy.comfy_types.node_typing import ComfyNodeABC
from comfy_api.latest import io


def v3_schema_stub(module: ModuleType) -> list[type[io.ComfyNode]]:
    NODE_CLASS_MAPPINGS: dict[str, type[ComfyNodeABC]] = module.NODE_CLASS_MAPPINGS
    NODE_DISPLAY_NAME_MAPPINGS: dict[str, str] = (
        module.NODE_DISPLAY_NAME_MAPPINGS if hasattr(module, "NODE_DISPLAY_NAME_MAPPINGS") else {}
    )

    @dataclass
    class SchemaPPMStub:
        node_id: str
        display_name: str | None

    def inject_schema_stub(cls: type[ComfyNodeABC], node_id: str, display_name: str | None = None):
        schema = SchemaPPMStub(node_id, display_name)

        if not hasattr(cls, "GET_SCHEMA"):
            setattr(cls, "GET_SCHEMA", lambda: schema)

        return cls

    return [inject_schema_stub(m[1], m[0], NODE_DISPLAY_NAME_MAPPINGS.get(m[0])) for m in NODE_CLASS_MAPPINGS.items()]  # type: ignore
