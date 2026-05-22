"""CAFA5 dataset helpers."""

import importlib
import sys
import types
from typing import Any


def __getattr__(name: str) -> Any:
    if name == "load":
        module_name = f"{__name__}.load"
        try:
            module = importlib.import_module(module_name)
        except ModuleNotFoundError as exc:
            if normalize_missing_module(exc) != "datasets":
                raise
            module = types.ModuleType(module_name)

            def load_cafa5_dataset(*args: Any, **kwargs: Any) -> Any:
                raise RuntimeError("Loading CAFA5 datasets requires the optional 'datasets' dependency.")

            module.load_cafa5_dataset = load_cafa5_dataset
            sys.modules[module_name] = module
        globals()[name] = module
        return module
    raise AttributeError(name)


def normalize_missing_module(exc: ModuleNotFoundError) -> str:
    return str(getattr(exc, "name", "") or "")


__all__ = ["load"]
