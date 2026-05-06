"""Mythos Transformer research prototype."""

__version__ = "1.0.0"
__author__ = "Your Organization"
__email__ = "team@your-org.com"

from .models.mythos_transformer import MythosTransformer
from .models.local_transformer import LocalTransformer

__all__ = ["MythosTransformer", "LocalTransformer"]
