"""Small deterministic tools for Persian."""

import importlib

__version__ = "0.2.0"
__all__ = ["arabic", "aruz", "ezafe", "orthography", "verb", "vowels"]


def __getattr__(name):
    if name in __all__:
        return importlib.import_module(f".{name}", __name__)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
