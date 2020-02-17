"""
    np_rw_buffer.buffer
    SeaLandAire Technologies
    @author: jengel

Numpy circular buffer to help store audio data.


"""
import numpy as np
import threading

from .utils import make_thread_safe
from .circular_indexes import get_indexes


__all__ = ["UnderflowError", "get_shape_columns", "get_shape", 'reshape', "RingBuffer", "RingBufferThreadSafe"]


UnderflowError = ValueError


def get_shape(shape):
    """Return rows, columns for the shape."""
    try:
        return (shape[0], shape[1]) + shape[2:]
    except IndexError:
        return (shape[0], 0) + shape[2:]
    except TypeError:
        return int(shape), 0
# get_shape


def get_shape_columns(shape):
    """Return the number of columns for the shape."""
    try:
        return shape[1]
    except (IndexError, TypeError):
        return 0
# end get_shape_columns


def reshape(ring_buffer, shape):
    """Safely reshape the data.

    Args:
        ring_buffer (RingBuffer/np.ndarray/np.array): Array to reshape
        shape (tuple): New shape
    """
    try:
        buffer = ring_buffer._data
    except AttributeError:
        buffer = ring_buffer

    new_shape = get_shape(shape)
    myshape = get_shape(buffer.shape)
    if new_shape[1] == 0:
        new_shape = (new_shape[0], 1) + new_shape[2:]

    if new_shape[0] == -1:
        try:  # Only change the column shape
            buffer.shape = new_shape
        except ValueError:  # Change the entire array shape
            rows = int(np.ceil(myshape[0]/new_shape[1]))
            new_shape = (rows, ) + new_shape[1:]
            buffer.resize(new_shape, refcheck=False)

    else:
        # Force proper sizing
        buffer.resize(new_shape, refcheck=False)

        # Clear the buffer if it did anything but grow in length
        # if not (new_shape[0] > myshape[0] and new_shape[1:] == myshape[1:]):
        try:
            ring_buffer.clear()
        except AttributeError:
            pass


def format_write_data(data, mydtype):
    """Format the given data to the proper shape that can be written into this buffer."""
    try:
        len(data)  # Raise TypeError if no len
        dshape = data.shape
    except TypeError:
        # Data has no length
        data = np.asarray(data, dtype=mydtype)
        dshape = data.shape

    except AttributeError:
        # Data is not a numpy array
        data = np.asarray(data, dtype=mydtype)
        dshape = data.shape

    # Force at least 1 column
    if get_shape_columns(dshape) == 0:
        data = np.reshape(data, (-1, 1))
    dshape = data.shape

    return data, dshape
# end format_write_data


class RingBuffer(object):
    """Numpy circular buffer to help store audio data.

    Args:
        shape (tuple/int): Length of the buffer.
        columns (int)[1]: Columns for the buffer.
        dtype (numpy.dtype)[numpy.float32]: Numpy data type for the buffer.
    """

    def __init__(self, shape, columns=None, dtype=np.float32):
        self._start = 0
        self._end = 0
        self._length = 0
        self.lock = threading.RLock()

        # Configure the shape (check if the given length was really the shape)
        if isinstance(shape, (tuple, list)):
            shape = shape
            if columns is not None and columns > 0:
                shape = (shape[0], columns) + shape[2:]
        else:
            if columns is None:
                columns = 1
            shape = (shape, columns)

        # Force columns
        if get_shape_columns(shape) == 0:
            shape = (shape[0], 1) + shape[2:]

        # Create the data buffer
        shape = tuple((int(np.ceil(i)) for i in shape))
        self._data = np.zeros(shape=shape, dtype=dtype)
    # end constructor

    def clear(self):
        """Clear the data."""
        self._start = 0
        self._end = 0
        self._length = 0
    # end clear

    def get_data(self):
        """Return the data in the buffer without moving the start pointer."""
        idxs = self.get_indexes(self._start, self._length, self.maxsize)
        return self._data[idxs].copy()
    # end get_data

    def set_data(self, data):
        """Set the data."""
        self.dtype = data.dtype
        self.shape = data.shape
        self.clear()
        self.expanding_write(data)
    # end set_data

    def _write(self, data, length, error, move_start=True):
        """Actually write the data to the numpy array.

        Args:
            data (np.array/np.ndarray): Numpy array of data to write. This should already be in the correct format.
            length (int): Length of data to write. (This argument needs to be here for error purposes).
            error (bool): Error on overflow else overrun the start pointer or move the start pointer to prevent
                overflow (Makes it circular).
            move_start (bool)[True]: If error is false should overrun occur or should the start pointer move.

        Raises:
            OverflowError: If error is True and more data is being written then there is space available.
        """
        idxs = self.get_indexes(self._end, length, self.maxsize)
        self.move_end(length, error, move_start)
        self._data[idxs] = data

    def expanding_write(self, data, error=True):
        """Write data into the buffer. If the data is larger than the buffer expand the buffer.

        Args:
            data (numpy.array): Data to write into the buffer.
            error (bool)[True]: Error on overflow else overrun the start pointer or move the start pointer to prevent
                overflow (Makes it circular).

        Raises:
            ValueError: If data shape does not match this shape. Arrays without a column will be convert to 1 column
                Example: (5,) will become (5, 1) and will not error if there is 1 column
            OverflowError: If the written data will overflow the buffer.
        """
        data, shape = format_write_data(data, self.dtype)

        length = shape[0]
        if shape[1:] != self.shape[1:]:
            msg = "could not broadcast input array from shape {:s} into shape {:s}".format(str(shape), str(self.shape))
            raise ValueError(msg)
        elif length > self.maxsize:
            self.shape = (length, ) + self.shape[1:]

        self._write(data, length, error)
    # end expanding_write

    def growing_write(self, data):
        """Write data into the buffer. If there is not enough available space then grow the buffer.

        Args:
            data (numpy.array): Data to write into the buffer.

        Raises:
            ValueError: If data shape does not match this shape. Arrays without a column will be convert to 1 column
                Example: (5,) will become (5, 1) and will not error if there is 1 column
            OverflowError: If the written data will overflow the buffer.
        """
        data, shape = format_write_data(data, self.dtype)

        length = shape[0]
        available = self.get_available_space()
        if shape[1:] != self.shape[1:]:
            msg = "could not broadcast input array from shape {:s} into shape {:s}".format(str(shape), str(self.shape))
            raise ValueError(msg)
        elif length > available:
            # Keep the old data and reshape
            old_data = self.get_data()
            self.shape = (self.maxsize + (length - available),) + self.shape[1:]
            if len(old_data) > 0:
                self._write(old_data, len(old_data), False)

        self._write(data, length, error=True)
    # end expanding_write

    def write(self, data, error=True):
        """Write data into the buffer.

        Args:
            data (numpy.array): Data to write into the buffer.
            error (bool)[True]: Error on overflow else overrun the start pointer or move the start pointer to prevent
                overflow (Makes it circular).

        Raises:
            ValueError: If data shape does not match this shape. Arrays without a column will be convert to 1 column
                Example: (5,) will become (5, 1) and will not error if there is 1 column
            OverflowError: If the written data will overflow the buffer.
        """
        data, shape = format_write_data(data, self.dtype)
        length = shape[0]
        if shape[1:] != self.shape[1:]:
            msg = "could not broadcast input array from shape {:s} into shape {:s}".format(str(shape), str(self.shape))
            raise ValueError(msg)
        elif not error and length > self.maxsize:
            data = data[-self.maxsize:]
            length = self.maxsize

        self._write(data, length, error)
    # end write

    def read(self, amount=None):
        """Read the data and move the start/read pointer, so that data is not read again.

        This method reads empty if the amount specified is greater than the amount in the buffer.

        Args:
            amount (int)[None]: Amount of data to read
        """
        if amount is None:
            amount = self._length

        # Check available read size
        if amount == 0 or amount > self._length:
            return self._data[0:0].copy()

        idxs = self.get_indexes(self._start, amount, self.maxsize)
        self.move_start(amount)
        return self._data[idxs].copy()
    # end read

    def read_remaining(self, amount=None):
        """Read the data and move the start/read pointer, so that the data is not read again.

        This method reads the remaining data if the amount specified is greater than the amount in the buffer.

        Args:
            amount (int)[None]: Amount of data to read
        """
        if amount is None or amount > self._length:
            amount = self._length

        # Check available read size
        if amount == 0:
            return self._data[0:0].copy()

        idxs = self.get_indexes(self._start, amount, self.maxsize)
        self.move_start(amount)
        return self._data[idxs].copy()
    # end read_remaining

    def read_overlap(self, amount=None, increment=None):
        """Read the data and move the start/read pointer.

        This method only increments the start/read pointer the given increment amount. This way the same data can be
        read multiple times.

        This method reads empty if the amount specified is greater than the amount in the buffer.

        Args:
            amount (int)[None]: Amount of data to read
            increment (int)[None]: Amount to move the start/read pointer allowing overlap if increment is less than the
                given amount.
        """
        if amount is None:
            amount = self._length
        if increment is None:
            increment = amount

        # Check available read size
        if amount == 0 or amount > self._length:
            return self._data[0:0].copy()

        idxs = self.get_indexes(self._start, amount, self.maxsize)
        self.move_start(increment)
        return self._data[idxs].copy()
    # end read_overlap

    def read_last(self, amount=None, update_rate=None):
        """Read the last amount of data and move the start/read pointer.

        This is an odd method for FFT calculations. It reads the newest data moving the start pointer by the
        update_rate amount that it was given. The returned skips number is the number of update_rate values.

        Example:

            .. code-block :: python

                >>> buffer = RingBuffer(11, 1)
                >>> buffer.write([0, 1, 2, 3, 4, 5, 6, 7 ,8, 9, 10])
                >>> buffer.read_last(6, 2))
                (array([[4.],
                        [5.],
                        [6.],
                        [7.],
                        [8.],
                        [9.]], dtype=float32), 3)
                >>> # Note must read in a multiple of the amount and moves by a multiple of the update rate.

        Args:
            amount (int)[None]: Amount of data to read. NFFT value.
            update_rate (int)[None]: The fft update rate value. How many samples to move the pointer by
                to cause overlap.

        Returns:
            data (np.array/np.ndarray) [None]: Data that is of length amount.
            updates (int) [0]: Number of updates (Total number of update rates until the end of the data was
                found including the data that was returned).
        """
        if amount is None:
            amount = self._length
        if update_rate is None:
            update_rate = amount

        # Check available read size
        if amount == 0 or amount > self._length:
            return None, 0

        skips = (self._length - amount) // update_rate
        if skips > 0:
            self.move_start(update_rate * skips)
        idxs = self.get_indexes(self._start, amount, self.maxsize)
        self.move_start(update_rate)
        return self._data[idxs].copy(), skips + 1
    # end read_last

    def __len__(self):
        """Return the current size of the buffer."""
        return self._length

    def __str__(self):
        return self.get_data().__str__()

    get_indexes = staticmethod(get_indexes)

    def move_start(self, amount, error=True, limit_amount=True):
        """This is an internal method and should not need to be called by the user.

        Move the start pointer the given amount (+/-).

        Raises:
            UnderflowError: If the amount is > the length.

        Args:
            amount (int): Amount to move the start pointer by.
            error (bool)[True]: Raise a ValueError else sync the end pointer and length.
            limit_amount (bool)[True]: If True force the amount to be less than or equal to the amount in the buffer.
        """
        if amount == 0:
            return
        elif amount > self._length:
            if error:
                raise UnderflowError("Not enough data in the buffer " + repr(self))

            if limit_amount:
                # You cannot read more than what you have
                amount = self._length
        # end error

        stop = self._start + amount
        try:
            self._start = stop % self.maxsize
        except ZeroDivisionError:
            self._start = stop

        self.sync_length(False or amount < 0)  # Length grows if amount was negative.
    # end move_start

    def move_end(self, amount, error=True, move_start=True):
        """This is an internal method and should not need to be called by the user.

        Move the end pointer the given amount (+/-).

        Raises:
            OverflowError: If the amount is > the available buffer space.

        Args:
            amount (int): Amount to move the end pointer by.
            error (bool)[True]: Raise an OverflowError else sync the start pointer and length.
            move_start (bool)[True]: If True and amount > available move the start pointer with the end pointer.
        """
        # Check for overflow
        avaliable = self.maxsize - self._length
        if amount == 0:
            return
        elif amount > 0 and amount > avaliable:
            if error:
                raise OverflowError("Not enough space in the buffer " + repr(self) +
                                    " " + repr(len(self)) + " < " + repr(amount))

            if move_start:
                # Move the start to make it a circular
                make_available = amount - avaliable
                self.move_start(make_available, False)  # Needs to move for sync_length
                if amount > self.maxsize:
                    self.move_start(-(amount - self.maxsize) - 1, False)  # Needs to move for sync_length

        stop = self._end + amount
        try:
            self._end = stop % self.maxsize
        except ZeroDivisionError:
            self._end = stop

        self.sync_length(True and amount >= 0)  # Length shrinks if amount was negative.
    # end move_end

    def sync_length(self, should_grow=True):
        """Sync the length with the start and end pointers.

        Args:
            should_grow (int): Determines if start and end equal means full or empty.
                Writing can make full, reading empty.
        """
        try:
            self._length = (self._end - self._start) % self.maxsize
        except ZeroDivisionError:
            self._length = 0

        if self._length == 0 and should_grow:
            self._length = self.maxsize
    # end sync_length

    @property
    def maxsize(self):
        """Return the maximum buffer size."""
        return len(self._data)

    @maxsize.setter
    def maxsize(self, maxsize):
        """Set the maximum size."""
        self.shape = (int(maxsize), ) + self.shape[1:]
        self.clear()

    def get_available_space(self):
        """Return the available space."""
        return self.maxsize - len(self)

    @property
    def columns(self):
        """Return the number of columns/columns."""
        try:
            return self._data.shape[1] or 1
        except (AttributeError, IndexError):
            return 1

    @columns.setter
    def columns(self, columns):
        """Set the columns."""
        self.shape = (self.maxsize, columns) + self.shape[2:]
        self.clear()

    @property
    def shape(self):
        """Return the shape of the data."""
        return self._data.shape

    @shape.setter
    def shape(self, new_shape):
        """Set the shape."""
        reshape(self, new_shape)

    @property
    def dtype(self):
        """Return the dtype of the data."""
        return self._data.dtype

    @dtype.setter
    def dtype(self, dtype):
        try:
            self._data = self._data.astype(dtype)
        except (AttributeError, ValueError, TypeError, Exception):
            self._data = np.zeros(shape=self.shape, dtype=dtype)
            self.clear()
# end class RingBuffer


class RingBufferThreadSafe(RingBuffer):
    """Standard numpy circular buffer.

    Args:
        length (tuple/int): Length of the buffer.
        columns (int)[1]: Columns for the buffer.
        dtype (numpy.dtype)[numpy.float32]: Numpy data type for the buffer.
    """
    def __init__(self, shape, columns=None, dtype=np.float32):
        self.lock = threading.RLock()
        super().__init__(shape=shape, columns=columns, dtype=dtype)
    # end constructor

    clear = make_thread_safe(RingBuffer.clear)
    get_data = make_thread_safe(RingBuffer.get_data)
    set_data = make_thread_safe(RingBuffer.set_data)

    expanding_write = make_thread_safe(RingBuffer.expanding_write)
    growing_write = make_thread_safe(RingBuffer.growing_write)
    write = make_thread_safe(RingBuffer.write)

    read = make_thread_safe(RingBuffer.read)
    read_remaining = make_thread_safe(RingBuffer.read_remaining)
    read_overlap = make_thread_safe(RingBuffer.read_overlap)
    read_last = make_thread_safe(RingBuffer.read_last)

    __len__ = make_thread_safe(RingBuffer.__len__)
    __str__ = make_thread_safe(RingBuffer.__str__)

    move_start = make_thread_safe(RingBuffer.move_start)
    move_end = make_thread_safe(RingBuffer.move_end)
    sync_length = make_thread_safe(RingBuffer.sync_length)

    get_available_space = make_thread_safe(RingBuffer.get_available_space)
    maxsize = make_thread_safe(RingBuffer.maxsize)
    columns = make_thread_safe(RingBuffer.columns)
    shape = make_thread_safe(RingBuffer.shape)
    dtype = make_thread_safe(RingBuffer.dtype)
