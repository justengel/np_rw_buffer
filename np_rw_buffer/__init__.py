from .circular_indexes import get_indexes
py_get_indexes = get_indexes

try:
    from ._circular_indexes import get_indexes
    USING_C = True
    c_get_indexes = get_indexes
except (ImportError, Exception):
    USING_C = False
    c_get_indexes = None

from .buffer import *
from .audio_buffer import *
from .manager import *
