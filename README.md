# Numpy Read Write Buffer

This library was created to help store audio data in a numpy array. It allows for writing lists and numpy arrays into a circular buffer. You can read the data from the buffer with overlap to perform smooth FFTs.
The buffer is a wrapper around a numpy ndarray. It contains a start and end position to keep track of where to read and write the data.

Main Functions:
  * clear() - Clear the length, start, and end indexes
  * get_data() - Return a copy of the data without moving indexes
  * set_data(data) - Set the data and change the shape of the buffer to this data shape
  * write(data, error) - Write data into the buffer and move the end index
  * read(amount) - Read data from the buffer and move the start index. If the amount is greater that what is in the buffer return a 0 length buffer

Extra Functions to help with the start and end pointers:
  * expanding_write(data, error) - Write data into the buffer. If the data is larger than the buffer expand the buffer
  * growing_write(data) - Write data into the buffer if there is not enough space make the buffer larger
  * read_remaining(amount) - Read the amount or read all of the data available to read
  * read_overlap(amount, increment) - Read the amount of data given, but only increment the start index by the increment amount. This makes the next read, read some duplicate data (hence overlap)

Buffer Control Functions:
  * maxsize - (property) change the amount of samples that can be held
  * columns - (property) Number of columns that the array contains (shape[1])
  * shape - (property) Change the shape of the buffer
  * dtype - (property) Change the data type for the numpy buffer
  * get_indexes(start, length) - Return a list of indexes for reading and writing (this makes the buffer circular)
  * move_start(amount, error) - Move the start index (read)
  * move_end(amount, error) - Move the end index (write)
  * get_available_space() - return the amount of data that the buffer can still hold


## Example - simple example
Simple reading and writing. See test_buffer for tests and usage.

```python
import numpy as np
import np_rw_buffer

buffer = np_rw_buffer.RingBuffer(10)

buffer.write(np.arange(5))
r = buffer.read(4)
assert np.all(r == np.arange(4).reshape((-1, 1)))

# Not enough data, don't read anything (use read_remaining or get_data)
d = np.arange(5).reshape((-1, 1))
buffer.write(d)
r = buffer.read(10)
assert len(r) == 0
assert len(buffer) == 6

r = buffer.read()
assert len(r) == 6
assert np.all(r == np.vstack((d[-1:], d)))

buffer.write(np.arange(6))
# buffer.write(np.arange(5))  # Raises an OverflowError
buffer.write(np.arange(5), False)
```


## Example - AudioFramingBuffer
The AudioFramingBuffer is slightly different from the RingBuffer. It has a sample_rate, seconds, and buffer_delay.

It's main differences are how it reads and writes. The start and end pointers are completely different and decoupled. 
The start pointer can underrun the end pointer and back fills with 0's. The end pointer can overrun the start pointer.

```python
import numpy as np
from np_rw_buffer import AudioFramingBuffer

buffer = AudioFramingBuffer(2000, 1)
buffer.write(np.array([(i,) for i in range(10)]))
# Buffer: [(read ptr)0, 1, 2, 3, 4, 5, 6, 7, 8, 9, (write ptr) 0, 0, 0, 0, 0]
assert buffer._end == 10
assert buffer._start == 0

# [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, (write ptr) 0, 0, 0, 0, 0] (read ptr at end)
assert np.all(buffer.read(15) == np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 0, 0, 0, 0, 0]).reshape((-1, 1)))
assert buffer._start == 15
assert buffer._end == 10

buffer.write(np.array([(i,) for i in range(10)])) # This will write in the position after 19
# Buffer: [0, 0, 0, 0, 0, 0, 0, 0, 0, (was 9) 0, 0, 1, 2, 3, 4, (read ptr) 5, 6, 7, 8, 9] (write ptr at end)
assert buffer._end == 20
assert buffer._start == 15

# [5, 6, 7, 8, 9, (write ptr) 0, 0, 0, 0, 0] (read ptr at end)
assert np.all(buffer.read(10) == np.array([5, 6, 7, 8, 9, 0, 0, 0, 0, 0]).reshape((-1, 1)))
```
