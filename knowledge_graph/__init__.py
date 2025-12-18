"""Knowledge Graph for LG Electronics HE (TV) Business Analysis

Layer 1: Dimension & Anchor (3NF RDB Schema)
Layer 2: Factor extraction from Consensus/Dart documents
Layer 3: Event mapping (future)
"""

from .config import BaseConfig

__all__ = [
    "BaseConfig",
]
