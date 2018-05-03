"""
    np_rw_buffer.buffer
    SeaLandAire Technologies
    @author: jengel

Numpy circular buffer to help store audio data.


"""
import numpy as np
import threading


__all__ = ["UnderflowError", "get_shape_columns", "get_shape", "RingBuffer", "RingBufferThreadSafe"]


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
        length (tuple/int): Length of the buffer.
        columns (int)[1]: Columns for the buffer.
        dtype (numpy.dtype)[numpy.float32]: Numpy data type for the buffer.
    """

    def __init__(self, length, columns=1, dtype=np.float32):
        self._start = 0
        self._end = 0
        self._length = 0
        self.lock = threading.RLock()

        # Configure the shape (check if the given length was really the shape)
        shape = (length, columns)
        if isinstance(length, (tuple, list)):
            shape = length
            if columns != 1 and columns > 0:
                shape = (shape[0], columns) + shape[2:]

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
        idxs = self.get_indexes(self._start, self._length)
        return self._data[idxs]
    # end get_data

    def set_data(self, data):
        """Set the data."""
        self.dtype = data.dtype
        self.shape = data.shape
        self.clear()
        self.expanding_write(data)
    # end set_data

    def expanding_write(self, data, error=True):
        """Write data into the buffer. If the data is larger than the buffer expand the buffer.

        Args:
            data (numpy.array): Data to write into the buffer.
            error (bool)[True]: Error on overflow else move the start pointer to prevent overflow (Makes it circular).

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

        idxs = self.get_indexes(self._end, length)
        self.move_end(length, error)
        self._data[idxs] = data
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
            self.write(old_data, False)

        idxs = self.get_indexes(self._end, length)
        self.move_end(length)
        self._data[idxs] = data
    # end expanding_write

    def write(self, data, error=True):
        """Write data into the buffer.

        Args:
            data (numpy.array): Data to write into the buffer.
            error (bool)[True]: Error on overflow else move the start pointer to prevent overflow (Makes it circular).

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

        idxs = self.get_indexes(self._end, length)
        self.move_end(length, error)
        self._data[idxs] = data
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
            return self._data[0:0]

        idxs = self.get_indexes(self._start, amount)
        self.move_start(amount)
        return self._data[idxs]
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
            return self._data[0:0]

        idxs = self.get_indexes(self._start, amount)
        self.move_start(amount)
        return self._data[idxs]
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
            return self._data[0:0]

        idxs = self.get_indexes(self._start, amount)
        self.move_start(increment)
        return self._data[idxs]
    # end read_overlap

    def __len__(self):
        """Return the current size of the buffer."""
        return self._length

    def __str__(self):
        return self.get_data().__str__()

    def get_indexes(self, start, length):
        """Return the indexes from the given start position to the given length."""
        stop = start + length

        # Check roll-over/roll-under
        if stop > self.maxsize:
            try:
                return np.concatenate((np.arange(start, self.maxsize),
                                       np.arange(0, stop % self.maxsize)))
            except ZeroDivisionError:
                return []
        elif stop < 0:
            return np.concatenate((np.arange(start, -1, -1),
                                   np.arange(self.maxsize, self.maxsize-stop, -1)))
        # get the step
        try:
            return slice(start, stop, length//abs(length))
        except ZeroDivisionError:
            return slice(start, stop)
    # end get_indexes

    def move_start(self, amount, error=True):
        """This is an internal method and should not need to be called by the user.

        Move the start pointer the given amount (+/-).

        Raises:
            ValueError: If the amount is > the length.

        Args:
            amount (int): Amount to move the start pointer by.
            error (bool)[True]: Raise a ValueError else sync the end pointer and length.
        """
        if amount > self._length:
            if error:
                raise UnderflowError("Not enough data in the buffer " + repr(self))

            # You cannot read more than what you have
            amount = self._length
        # end error

        stop = self._start + amount
        try:
            self._start = stop % self.maxsize
        except ZeroDivisionError:
            self._start = stop

        self.sync_length(False)
    # end move_start

    def move_end(self, amount, error=True):
        """This is an internal method and should not need to be called by the user.

        Move the end pointer the given amount (+/-).

        Raises:
            OverflowError: If the amount is > the available buffer space.

        Args:
            amount (int): Amount to move the end pointer by.
            error (bool)[True]: Raise an OverflowError else sync the start pointer and length.
        """
        # Check for overflow
        avaliable = self.maxsize - self._length
        if amount > avaliable:
            if error:
                raise OverflowError("Not enough space in the buffer " + repr(self) +
                                    " " + repr(len(self)) + " < " + repr(amount))

            # Move the start to make it a circular
            make_available = amount - avaliable
            self.move_start(make_available, False)
            if amount > self.maxsize:
                self.move_start(-(amount - self.maxsize) - 1, False)

        stop = self._end + amount
        try:
            self._end = stop % self.maxsize
        except ZeroDivisionError:
            self._end = stop

        self.sync_length(True)
    # end move_end

    def sync_length(self, is_write=True):
        """Sync the length with the start and end pointers.

        Args:
            is_write (int): Determines if start and end equal means full or empty. Writing can make full, reading empty.
        """
        try:
            self._length = (self._end - self._start) % self.maxsize
        except ZeroDivisionError:
            self._length = 0

        if self._length == 0 and is_write:
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
        new_shape = get_shape(new_shape)
        myshape = get_shape(self.shape)
        if new_shape[1] == 0:
            new_shape = (new_shape[0], 1) + new_shape[2:]

        if new_shape[0] == -1:
            try:  # Only change the column shape
                self._data.shape = new_shape
            except ValueError:  # Change the entire array shape
                rows = int(np.ceil(myshape[0]/new_shape[1]))
                new_shape = (rows, ) + new_shape[1:]
                self._data.resize(new_shape, refcheck=False)

        else:
            # Force proper sizing
            self._data.resize(new_shape, refcheck=False)

            # Clear the buffer if it did anything but grow in length
            # if not (new_shape[0] > myshape[0] and new_shape[1:] == myshape[1:]):
            self.clear()
    # end shape

    @property
    def dtype(self):
        """Return the dtype of the data."""
        return self._data.dtype

    @dtype.setter
    def dtype(self, dtype):
        try:
            self._data = self._data.astype(dtype)
        except (AttributeError, ValueError, TypeError):
            self._data = np.zeros(shape=self.shape, dtype=dtype)
            self.clear()
# end class RingBuffer


def make_thread_safe(lock_varname="lock"):
    """Decorate a function making it threadsafe by using the threading lock that matches the lock_varname.

    Args:
        lock_varname (str/method)['lock']: Threading lock variable name or
            a function to decorate with 'lock' variable being a threading.Lock

    Returns:
        decorator (function): Function that was decorated or a function that will decorate a function.
    """
    if not isinstance(lock_varname, str):
        # Function was given decorate the function
        func = lock_varname
        lock_varname = 'lock'

        def wrapper(*args, **kwargs):
            with getattr(args[0], lock_varname):
                return func(*args, **kwargs)
        return wrapper
    else:
        # lock_varname was given return the real decorator
        def real_decorator(func):
            def wrapper(*args, **kwargs):
                with getattr(args[0], lock_varname):
                    return func(*args, **kwargs)
            return wrapper
        return real_decorator


class RingBufferThreadSafe(RingBuffer):
    """Standard numpy circular buffer

    Args:
        length (tuple/int): Length of the buffer.
        columns (int)[1]: Columns for the buffer.
        dtype (numpy.dtype)[numpy.float32]: Numpy data type for the buffer.
    """
    def __init__(self, length, columns=1, dtype=np.float32):
        self.lock = threading.RLock()
        super().__init__(length=length, columns=columns, dtype=dtype)
    # end constructor

    @make_thread_safe()
    def clear(self):
        """Clear the data."""
        self._start = 0
        self._end = 0
        self._length = 0
    # end clear

    @make_thread_safe()
    def get_data(self):
        """Return the data in the buffer without moving the start pointer."""
        idxs = self.get_indexes(self._start, self._length)
        return self._data[idxs]
    # end get_data

    @make_thread_safe()
    def set_data(self, data):
        """Set the data."""
        self.dtype = data.dtype
        self.shape = data.shape
        self.clear()
        self.expanding_write(data)
    # end set_data

    @make_thread_safe()
    def expanding_write(self, data, error=True):
        """Write data into the buffer. If the data is larger than the buffer expand the buffer.

        Args:
            data (numpy.array): Data to write into the buffer.
            error (bool)[True]: Error on overflow else move the start pointer to prevent overflow (Makes it circular).
        """
        data, shape = format_write_data(data, self.dtype)

        length = shape[0]
        if length > self.maxsize:
            self.shape = (length, ) + self.shape[1:]

        idxs = self.get_indexes(self._end, length)
        self.move_end(length, error)
        self._data[idxs] = data
    # end expanding_write

    @make_thread_safe()
    def growing_write(self, data):
        """Write data into the buffer. If there is not enough available space then grow the buffer.

        Args:
            data (numpy.array): Data to write into the buffer.
        """
        data, shape = format_write_data(data, self.dtype)

        length = shape[0]
        if length > self.get_available_space():
            self.shape = (self.maxsize + length, ) + self.shape[1:]

        idxs = self.get_indexes(self._end, length)
        self.move_end(length)
        self._data[idxs] = data
    # end expanding_write

    @make_thread_safe()
    def write(self, data, error=True):
        """Write data into the buffer.

        Args:
            data (numpy.array): Data to write into the buffer.
            error (bool)[True]: Error on overflow else move the start pointer to prevent overflow (Makes it circular).
        """
        data, shape = format_write_data(data, self.dtype)
        length = shape[0]

        idxs = self.get_indexes(self._end, length)
        self.move_end(length, error)
        self._data[idxs] = data
    # end write

    @make_thread_safe()
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
            return self._data[0:0]

        idxs = self.get_indexes(self._start, amount)
        self.move_start(amount)
        return self._data[idxs]
    # end read

    @make_thread_safe()
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
            return self._data[0:0]

        idxs = self.get_indexes(self._start, amount)
        self.move_start(amount)
        return self._data[idxs]
    # end read_remaining

    @make_thread_safe()
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
            return self._data[0:0]

        idxs = self.get_indexes(self._start, amount)
        self.move_start(increment)
        return self._data[idxs]
    # end read_overlap

    @make_thread_safe()
    def read_last(self, amount=None, update_rate=None):
        """Read the last amount of data and move the start/read pointer.

        This is an odd method for FFT calculations.

        amount is the amount of data returned.
        increment indicates where to the move the

        Args:
            amount (int)[None]: Amount of data to read. NFFT value.
            update_rate (int)[None]: The fft update rate value
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
        idxs = self.get_indexes(self._start, amount)
        self.move_start(update_rate)
        return self._data[idxs], skips + 1
    # end read_last

    @make_thread_safe()
    def __len__(self):
        """Return the current size of the buffer."""
        return self._length

    @make_thread_safe()
    def __str__(self):
        return self.get_data().__str__()

    @make_thread_safe()
    def get_indexes(self, start, length):
        """Return the indexes from the given start position to the given length."""
        stop = start + length

        # Check roll-over/roll-under
        if stop > self.maxsize:
            try:
                return np.concatenate((np.arange(start, self.maxsize),
                                       np.arange(0, stop % self.maxsize)))
            except ZeroDivisionError:
                return []
        elif stop < 0:
            return np.concatenate((np.arange(start, -1, -1),
                                   np.arange(self.maxsize, self.maxsize-stop, -1)))
        # get the step
        try:
            return slice(start, stop, length//abs(length))
        except ZeroDivisionError:
            return slice(start, stop)
    # end get_indexes

    @make_thread_safe()
    def move_start(self, amount, error=True):
        """This is an internal method and should not need to be called by the user.

        Move the start pointer the given amount (+/-).

        Raises:
            ValueError: If the amount is > the length.

        Args:
            amount (int): Amount to move the start pointer by.
            error (bool)[True]: Raise a ValueError else sync the end pointer and length.
        """
        if amount > self._length:
            if error:
                raise UnderflowError("Not enough data in the buffer (" +
                                     repr(amount) + ", " + repr(self._length) + ") " + repr(self))

            # You cannot read more than what you have
            amount = self._length
        # end error

        stop = self._start + amount
        try:
            self._start = stop % self.maxsize
        except ZeroDivisionError:
            self._start = stop

        self.sync_length(False)
    # end move_start

    @make_thread_safe()
    def move_end(self, amount, error=True):
        """This is an internal method and should not need to be called by the user.

        Move the end pointer the given amount (+/-).

        Raises:
            OverflowError: If the amount is > the available buffer space.

        Args:
            amount (int): Amount to move the end pointer by.
            error (bool)[True]: Raise an OverflowError else sync the start pointer and length.
        """
        # Check for overflow
        dist = self.maxsize - self._length
        if amount > dist:
            if error:
                raise OverflowError("Not enough space in the buffer " + repr(self))

            # Move the start to make it a circular buffer
            self.move_start((amount-dist), False)

        stop = self._end + amount
        try:
            self._end = stop % self.maxsize
        except ZeroDivisionError:
            self._end = stop

        self.sync_length(True)
    # end move_end

    @make_thread_safe()
    def sync_length(self, is_write=True):
        """Sync the length with the start and end pointers.

        Args:
            is_write (int): Determines if start and end equal means full or empty. Writing can make full, reading empty.
        """
        try:
            self._length = (self._end - self._start) % self.maxsize
        except ZeroDivisionError:
            self._length = 0

        if self._length == 0 and is_write:
            self._length = self.maxsize
    # end sync_length

    @property
    def maxsize(self):
        """Return the maximum buffer size."""
        with self.lock:
            return len(self._data)

    @maxsize.setter
    def maxsize(self, maxsize):
        """Set the maximum size."""
        with self.lock:
            self.shape = (maxsize, ) + self.shape[1:]
            self.clear()

    @make_thread_safe()
    def get_available_space(self):
        """Return the available space."""
        return self.maxsize - len(self)

    @property
    def columns(self):
        """Return the number of columns/columns."""
        with self.lock:
            try:
                return self._data.shape[1] or 1
            except (AttributeError, IndexError):
                return 1

    @columns.setter
    def columns(self, columns):
        """Set the columns."""
        with self.lock:
            self.shape = (self.maxsize, columns) + self.shape[2:]
            self.clear()

    @property
    def shape(self):
        """Return the shape of the data."""
        with self.lock:
            return self._data.shape

    @shape.setter
    def shape(self, new_shape):
        """Set the shape."""
        with self.lock:
            new_shape = get_shape(new_shape)
            myshape = get_shape(self.shape)
            if new_shape[1] == 0:
                new_shape = (new_shape[0], 1) + new_shape[2:]

            if new_shape[0] == -1:
                try:  # Only change the column shape
                    self._data.shape = new_shape
                except ValueError:  # Change the entire array shape
                    rows = int(np.ceil(myshape[0]/new_shape[1]))
                    new_shape = (rows, ) + new_shape[1:]
                    self._data.resize(new_shape, refcheck=False)

            else:
                # Force proper sizing
                self._data.resize(new_shape, refcheck=False)

                # Clear the buffer if it did anything but grow in length
                # if not (new_shape[0] > myshape[0] and new_shape[1:] == myshape[1:]):
                self.clear()
    # end shape

    @make_thread_safe()
    def reshape_and_zero(self, shape):
        """Reshape and add zeros to fill the structure."""
        self.shape = shape
        self.clear()
        self._data[:] = np.zeros(shape=shape, dtype=self.dtype)

    @property
    def dtype(self):
        """Return the dtype of the data."""
        with self.lock:
            return self._data.dtype

    @dtype.setter
    def dtype(self, dtype):
        with self.lock:
            try:
                self._data = self._data.astype(dtype)
            except (AttributeError, ValueError, TypeError):
                self._data = np.zeros(shape=self.shape, dtype=dtype)
                self.clear()
# end class RingBufferThreadSafe
