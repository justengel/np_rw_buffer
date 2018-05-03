import numpy as np
import np_rw_buffer


def test_buffer_control():
    buffer = np_rw_buffer.RingBuffer(10)

    maxs, col, l, dt = 10, 1, 0, np.float32
    assert buffer.maxsize == maxs, 'Incorrect maxsize - ' + ' expected '.join((str(buffer.maxsize), str(maxs)))
    assert buffer.columns == col, 'Incorrect columns - ' + ' expected '.join((str(buffer.maxsize), str(col)))
    assert list(buffer.shape) == [maxs, col], 'Incorrect buffer shape - ' + ' expected '.join((str(list(buffer.shape)),
                                                                                               str([maxs, col])))
    assert len(buffer) == l, 'Incorrect buffer length - ' + ' expected '.join((str(len(buffer)), str(l)))
    assert buffer.dtype == dt, 'Incorrect buffer dtype - ' + ' expected '.join((str(buffer.dtype), str(dt)))

    # ===== Test writing some data =====
    buffer.write(np.arange(5))
    assert len(buffer) == 5, 'Incorrect buffer length, is ' + ' should be '.join((str(len(buffer)), str(5)))
    assert buffer.get_available_space() == maxs-5, 'Incorrect available space'

    # Test overwriting the buffer with no error - len should be maxsize
    buffer.write(np.arange(15), False)  # Write without error overwriting some of the data
    assert len(buffer) == maxs, 'Incorrect buffer length, is ' + ' should be '.join((str(len(buffer)), str(maxs)))

    # ===== Test reshaping the buffer =====
    buffer.shape = (5, 2)
    maxs, col, l, dt = 5, 2, 0, np.float32
    assert buffer.maxsize == maxs, 'Incorrect maxsize - ' + ' expected '.join((str(buffer.maxsize), str(maxs)))
    assert buffer.columns == col, 'Incorrect columns - ' + ' expected '.join((str(buffer.maxsize), str(col)))
    assert list(buffer.shape) == [maxs, col], 'Incorrect buffer shape - ' + ' expected '.join((str(list(buffer.shape)),
                                                                                               str([maxs, col])))
    assert len(buffer) == l, 'Incorrect buffer length - ' + ' expected '.join((str(len(buffer)), str(l)))
    assert buffer.dtype == dt, 'Incorrect buffer dtype - ' + ' expected '.join((str(buffer.dtype), str(dt)))


def test_move_start_end():
    buffer = np_rw_buffer.RingBuffer(5, 2)
    maxs, col, l, dt = 5, 2, 0, np.float32

    # ===== Test moving the end (Fake writing) =====
    buffer.move_end(2)
    assert len(buffer) == 2
    buffer.move_end(3)
    assert len(buffer) == 5
    try:
        buffer.move_end(2)
        raise AssertionError('move_end should have caused an OverflowError')
    except OverflowError:
        pass

    # Check moving the end past the start (Overflow)
    assert len(buffer) == maxs
    buffer.move_end(2, False)  # Move past the start, start should move as well skipping over some unread data.
    assert len(buffer) == maxs, str(len(buffer)) + ' should be ' + str(maxs)
    buffer.move_end(3, False)  # Move past the start, start should move as well skipping over some unread data.
    assert len(buffer) == maxs, str(len(buffer)) + ' should be ' + str(maxs)
    buffer.move_end(1, False)  # Move past the start, start should move as well skipping over some unread data.
    assert len(buffer) == maxs, str(len(buffer)) + ' should be ' + str(maxs)
    buffer.move_end(maxs, False)  # Move past the start, start should move as well skipping over some unread data.
    assert len(buffer) == maxs, str(len(buffer)) + ' should be ' + str(maxs)
    buffer.move_end(maxs + 2, False)  # Move larger than the maxsize
    assert len(buffer) == maxs, str(len(buffer)) + ' should be ' + str(maxs)

    # ===== Test moving the start (Fake reading) =====
    buffer.move_start(2)
    assert len(buffer) == 3
    buffer.move_start(3)
    assert len(buffer) == 0
    try:
        buffer.move_start(4)
        raise AssertionError('move_start should have caused an UnderflowError')
    except np_rw_buffer.UnderflowError:
        pass
    buffer.move_start(3, False)
    assert len(buffer) == 0

    # ===== Test the clear function =====
    buffer.move_end(2)
    buffer.clear()
    assert len(buffer) == 0

    # ===== Test Reading and Writing Sync Length =====
    buffer.clear()
    assert len(buffer) == 0
    buffer.move_end(maxs)  # start 0, end 0?  Don't need to test exact positions just the functionality
    assert len(buffer) == maxs
    buffer.move_start(2)
    assert len(buffer) == maxs - 2
    buffer.move_start(len(buffer))
    assert len(buffer) == 0
    buffer.move_end(2)
    assert len(buffer) == 2
    buffer.move_start(2)
    assert len(buffer) == 0


def test_read_write():
    buffer = np_rw_buffer.RingBuffer(10, 2)
    maxs, col, l, dt = 10, 2, 0, np.float32

    # ===== Test Write =====
    d = np.array([[i, i*i] for i in range(3)])
    buffer.write(d)
    assert len(buffer) == len(d)

    # ===== Test Read =====
    r = buffer.read()
    assert np.all(r == d)
    assert len(buffer) == 0

    # ===== Test write and read when positions have moved =====
    d = np.array([[i, i*i] for i in range(4)])
    buffer.write(d)
    assert len(buffer) == len(d)

    # ===== Test read partial =====
    r = buffer.read(3)
    assert np.all(r == d[:3])
    assert len(buffer) == 1

    # ===== Test write wrap around (circular) =====
    buffer.write(d)
    assert len(buffer) == len(d) + 1

    # ===== Test write available space =====
    # Buffer will have d[-1] + d + d2[: 5]
    d2 = np.array([[(i+i), (i+i)*i] for i in range(8)])
    buffer.write(d2[:buffer.get_available_space()])
    assert len(buffer) == buffer.maxsize
    assert len(buffer) == maxs

    # ===== Test read all data =====
    r = buffer.read()
    assert np.all(r == np.vstack((d[-1:], d, d2[:5])))
    assert len(buffer) == 0

    # ===== Test write overflow error =====
    buffer.write(d)
    assert np.all(buffer.get_data() == d)
    try:
        buffer.write(d2)
        raise AssertionError("Should have caused an overflow error")
    except OverflowError:
        pass
    assert np.all(buffer.get_data() == d), "Data changed on OverflowError!"

    # ===== Test write overflow error ignored =====
    buffer.write(d2, False)
    assert np.all(buffer.get_data() == np.vstack((d[-2:], d2)))

    # ===== Test read too much outside of maxsize =====
    length = len(buffer)
    r = buffer.read(100)
    assert len(r) == 0
    assert len(buffer) == length

    # ===== Test read too much =====
    r = buffer.read(len(buffer) + 1)
    assert len(r) == 0
    assert len(buffer) == length

    # ===== Test read partial =====
    r = buffer.read(length - 2)
    assert len(buffer) == 2
    assert len(r) == length - 2
    assert np.all(r == np.vstack((d[-2:], d2))[:-2])


def test_expanding_write():
    """Expanding write make it so that write data that is larger than the entire buffer will change the size of the
    buffer to the new data size.
    """
    buffer = np_rw_buffer.RingBuffer(5, 1)
    maxs, col, l, dt = 5, 1, 0, np.float32

    # ===== Test writing data where size is > maxsize =====
    d = np.array([[i] for i in range(10)])
    buffer.expanding_write(d)
    assert buffer.maxsize == 10
    assert np.all(buffer.read() == d)

    # ===== Test writing data where the shape is different =====
    d = np.array([[i, i] for i in range(10)])
    try:
        buffer.expanding_write(d, error=False)
        raise AssertionError('This should error on a bad shape.')
    except ValueError as error:
        pass

    # ===== Test writing data overflow =====
    buffer.shape = (5, 1)
    d = np.array([[i] for i in range(3)])
    buffer.expanding_write(d)
    try:
        buffer.expanding_write(d)
        raise AssertionError("This should error. This is not a growing write call.")
    except OverflowError:
        pass
    buffer.expanding_write(d, False)
    assert len(buffer) == buffer.maxsize
    assert np.all(buffer.read() == np.vstack((d, d))[-5:])

    # ===== Test expanding with a different shape =====
    buffer.shape = (5, 1)
    d = np.array([[i, i] for i in range(8)])
    try:
        buffer.expanding_write(d)
        raise AssertionError('This should have error. Expanding still should not change shape')
    except ValueError:
        pass
    assert list(buffer.shape) == [5, 1]


def test_growing_write():
    buffer = np_rw_buffer.RingBuffer(5, 1)
    maxs, col, l, dt = 5, 1, 0, np.float32

    d = np.array([[i] for i in range(3)])
    buffer.growing_write(d)
    assert len(buffer) == 3

    buffer.growing_write(d)
    assert buffer.maxsize == 6, str(buffer.maxsize)
    assert len(buffer) == 6
    assert np.all(buffer.get_data() == np.vstack((d, d)))

    buffer.growing_write(d)
    assert buffer.maxsize == 9, str(buffer.maxsize)
    assert len(buffer) == 9, len(buffer)
    assert np.all(buffer.get_data() == np.vstack((d, d, d)))

    r = buffer.read(2)  # Check moved indexes grow (wrap around)
    buffer.growing_write(d)
    assert buffer.maxsize == 10, str(buffer.maxsize)
    assert len(buffer) == 10, len(buffer)
    assert np.all(buffer.get_data() == np.vstack((d[-1:], d, d, d)))


def test_read_remaining():
    buffer = np_rw_buffer.RingBuffer(10, 1)
    maxs, col, l, dt = 10, 1, 0, np.float32

    d = np.arange(10).reshape((-1, 1))
    buffer.write(d)
    assert len(buffer) == 10

    # Normal read does not read if the amount requested does not match the amount that it contains
    r = buffer.read(20)
    assert len(r) == 0
    assert len(buffer) == 10

    # Test read remaining to read all available data
    r = buffer.read_remaining(20)
    assert len(r) == 10
    assert len(buffer) == 0
    assert np.all(r == d)

    # Test normal read
    d = np.arange(3).reshape((-1, 1))
    buffer.write(d)
    r = buffer.read_remaining(2)
    assert len(r) == 2
    assert np.all(r == d[:2])

    # Test read remaining not outside maxsize
    buffer.write(d)
    r = buffer.read_remaining(5)
    assert len(r) == 4
    assert np.all(r == np.vstack((d[-1:], d)))


def test_read_overlap():
    buffer = np_rw_buffer.RingBuffer(10, 1)
    maxs, col, l, dt = 10, 1, 0, np.float32

    d = np.arange(10).reshape((-1, 1))
    buffer.write(d)
    assert len(buffer) == 10

    r = buffer.read_overlap(len(d), 2)
    assert len(r) == len(d)
    assert len(buffer) == len(d) - 2
    assert np.all(r == d)

    # This will also test wrap around
    d2 = np.arange(2).reshape((-1, 1))
    buffer.write(d2)
    r = buffer.read_overlap(10, 2)
    assert len(r) == 10
    assert len(buffer) == 10-2
    assert np.all(r == np.vstack((d[2:], d2)))


if __name__ == '__main__':
    test_buffer_control()
    test_move_start_end()
    test_read_write()
    test_expanding_write()
    test_growing_write()
    test_read_remaining()
    test_read_overlap()
    print('All tests finished successfully!')
