import numpy as np


def test_memory_manager():
    import np_rw_buffer

    shape = (10, 10)
    buf = np_rw_buffer.RingBuffer(shape)
    assert buf.shape == shape

    mngr = np_rw_buffer.MemoryManager(buf)
    assert mngr.shape == shape

    mngr.set_active(False)
    assert mngr.shape == shape
    assert buf.shape == (0,) + shape[1:]

    mngr.set_active(True)
    assert mngr.shape == shape
    assert buf.shape == shape

    # Test change shape while inactive
    mngr.set_active(False)
    assert mngr.shape == shape
    assert buf.shape == (0,) + shape[1:]

    mngr.shape = (5, 5)
    assert mngr.shape == (5, 5)
    assert buf.shape == (0,) + shape[1:]

    mngr.set_active(True)
    assert mngr.shape == (5, 5)
    assert buf.shape == (5, 5)


def test_buffer_control():
    import np_rw_buffer
    buffer = np_rw_buffer.MemoryManager(np_rw_buffer.RingBuffer(10))

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
    import np_rw_buffer
    buffer = np_rw_buffer.MemoryManager(np_rw_buffer.RingBuffer(5, 2))
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
    import np_rw_buffer
    buffer = np_rw_buffer.MemoryManager(np_rw_buffer.RingBuffer(10, 2))
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


if __name__ == '__main__':
    test_memory_manager()
    test_buffer_control()
    test_move_start_end()
    test_read_write()
    print('All tests finished successfully!')
