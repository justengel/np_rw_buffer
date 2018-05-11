import numpy as np
from np_rw_buffer import AudioFramingBuffer


def test_read_write():
    """Just use the old unit tests."""
    sf = 10
    sec = 2
    buffer = AudioFramingBuffer(sf, seconds=sec, buffer_delay=1)
    assert buffer.maxsize == sf * sec
    assert buffer.seconds == sec
    assert np.all(buffer.read(10) == np.zeros(10).reshape((-1, 1)))
    assert buffer.can_read is False

    buffer.write(np.arange(5))  # Wait until sf * buffer_delay (10)
    assert buffer.can_read is False
    assert np.all(buffer.read(10) == np.zeros(10).reshape((-1, 1)))

    buffer.write(np.arange(5))
    assert buffer.can_read is True
    assert np.all(buffer.read(10) == np.vstack((np.arange(5).reshape((-1, 1)),
                                                np.arange(5).reshape((-1, 1)))))


def test_example():
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


if __name__ == '__main__':
    test_read_write()
    test_example()
    print('All tests finished successfully!')
