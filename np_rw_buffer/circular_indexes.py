"""
Fast method to get the proper indexes for a circular buffer.

See history of version control. Python C tuple '_circular_indexes.c' building a tuple was much slower.
"""
import numpy as np


__all__ = ['get_indexes']


def get_indexes(start, length, maxsize):
    """Return the indexes from the given start position to the given length."""
    stop = start + length

    if stop > maxsize:
        # Check roll-over/roll-under
        try:
            return np.concatenate((np.arange(start, maxsize),
                                   np.arange(0, stop % maxsize)))
        except ZeroDivisionError:
            return []
    elif stop < 0:
        # Negative length roll-under
        return np.concatenate((np.arange(start, -1, -1),
                               np.arange(maxsize, maxsize-stop, -1)))

    # Return a simple slice
    try:
        return slice(start, stop, length//abs(length))
    except ZeroDivisionError:
        return slice(start, stop)
# end get_indexes
