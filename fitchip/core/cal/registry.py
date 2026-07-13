"""Backend discovery.

Backends self-register through the `fitchip.backends` entry-point group, so a
third-party adapter installed as a separate package (e.g. fitchip-backend-tvm)
appears here without the core importing it by name.
"""

from __future__ import annotations

from importlib.metadata import entry_points

from fitchip.core.cal.backend import CompilerBackend

ENTRY_POINT_GROUP = "fitchip.backends"


class BackendRegistry:
    def __init__(self) -> None:
        self._backends: dict[str, CompilerBackend] = {}
        for ep in entry_points(group=ENTRY_POINT_GROUP):
            backend_cls = ep.load()
            self._backends[ep.name] = backend_cls()
        # Built-ins ship in this same wheel, so make sure they are present even
        # when the installed entry-point metadata is missing or predates a new
        # backend (e.g. a stale editable install). Third-party backends still
        # arrive exclusively through the entry-point group.
        builtins = {
            "tflm": "fitchip.backends.tflm.adapter:TflmBackend",
            "executorch": "fitchip.backends.executorch.adapter:ExecutorchBackend",
        }
        for name, ref in builtins.items():
            if name not in self._backends:
                module_name, _, class_name = ref.partition(":")
                module = __import__(module_name, fromlist=[class_name])
                self._backends[name] = getattr(module, class_name)()

    def all(self) -> list[CompilerBackend]:
        return list(self._backends.values())

    def get(self, backend_id: str) -> CompilerBackend:
        try:
            return self._backends[backend_id]
        except KeyError:
            raise KeyError(
                f"Unknown backend '{backend_id}'. "
                f"Installed backends: {', '.join(sorted(self._backends))}"
            ) from None

    def ids(self) -> list[str]:
        return sorted(self._backends)
