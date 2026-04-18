__all__ = [
    "RosstatLoader", "NTDLoader", "CTALoader", "FileLoader",
    "Preprocessor",
    "SyntheticGenerator",
]


def __getattr__(name):
    if name in ("RosstatLoader", "NTDLoader", "CTALoader", "FileLoader"):
        from .loader import RosstatLoader, NTDLoader, CTALoader, FileLoader
        return locals()[name]
    if name == "Preprocessor":
        from .preprocessor import Preprocessor
        return Preprocessor
    if name == "SyntheticGenerator":
        from .synthetic import SyntheticGenerator
        return SyntheticGenerator
    raise AttributeError(f"module 'data' has no attribute {name!r}")
