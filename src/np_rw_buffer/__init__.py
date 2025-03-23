from .circular_indexes import get_indexes
py_get_indexes = get_indexes

try:
    from ._circular_indexes import get_indexes
    USING_C = True
    c_get_indexes = get_indexes
except (ImportError, Exception):
    USING_C = False
    c_get_indexes = None

from .buffer import UnderflowError, get_shape_columns, get_shape, reshape, RingBuffer, RingBufferThreadSafe
from .audio_buffer import UnderflowError, AudioFramingBuffer
from .manager import MemoryManager
