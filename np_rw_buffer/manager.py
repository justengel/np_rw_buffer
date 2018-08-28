from .buffer import reshape

__all__ = ['MemoryManager']


class MemoryManager(object):
    def __init__(self, buffer):
        self.free_memory = True  # Free memory when this plot is not active
        self._active = True  # Indicate that the plotting is active

        self._real_buffer = buffer
        self._shape = buffer.shape

    def is_active(self):
        """Return if the plot is active."""
        return self._active

    def set_active(self, active):
        """Set the plot to be active or inactive.

        This function also utilizes the memory manager to call functions to allocate or deallocate memory.
        """
        self._active = active

        if self.free_memory:
            if self.is_active():
                self._real_buffer.shape = self._shape
            else:
                self._real_buffer.shape = (0, ) + self._shape[1:]

    @property
    def shape(self):
        """Return the shape of the data."""
        return self._shape

    @shape.setter
    def shape(self, shape):
        """Set the shape of the data."""
        self._shape = shape
        if self.is_active() or not self.free_memory:
            reshape(self._real_buffer, shape)

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
        if self.is_active() or not self.free_memory:
            self._real_buffer._write(data, length, error, move_start=move_start)

    def __getattr__(self, item):
        return getattr(self._real_buffer, item)

    def __setattr__(self, key, value):
        if key in ['free_memory', '_active', '_real_buffer', '_shape', 'shape']:
            return super().__setattr__(key, value)
        return self._real_buffer.__setattr__(key, value)

    def __getitem__(self, item):
        return self._real_buffer.__getitem__(item)

    def __setitem__(self, key, value):
        if self.free_memory and not self.is_active():
            pass  # Do nothing?
        else:
            self._real_buffer.__setitem__(key, value)

    def __len__(self):
        return self._real_buffer.__len__()

    def __str__(self):
        return self._real_buffer.__str__()

    def __repr__(self):
        return self._real_buffer.__repr__()
