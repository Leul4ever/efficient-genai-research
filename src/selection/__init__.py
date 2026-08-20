from selection.base import REGISTRY, Selector, build, register  # noqa: F401
from selection import methods  # noqa: F401  (import registers the selectors)

__all__ = ["REGISTRY", "Selector", "build", "register"]
