"""Collection of helper utilities combining text/io/misc."""
from .text import normalize
from .io import read_text
from .misc import noop

__all__ = ["normalize", "read_text", "noop"]
