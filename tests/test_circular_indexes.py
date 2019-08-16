
def test_get_indexes():
    import timeit
    import numpy as np
    from np_rw_buffer.circular_indexes import get_indexes as py_get_indexes
    from np_rw_buffer._circular_indexes import get_indexes as c_get_indexes

    assert py_get_indexes(0, 10, 100) == c_get_indexes(0, 10, 100)  # Slice
    assert np.all(py_get_indexes(5, 10, 12) == c_get_indexes(5, 10, 12))
    assert py_get_indexes(0, 1000, 1000) == c_get_indexes(0, 1000, 1000)  # Slice
    assert np.all(py_get_indexes(500, 1000, 1000) == c_get_indexes(500, 1000, 1000))
    assert np.all(py_get_indexes(700, 1000, 1000) == c_get_indexes(700, 1000, 1000))


def time_get_indexes():
    import timeit
    from np_rw_buffer.circular_indexes import get_indexes as py_get_indexes
    from np_rw_buffer._circular_indexes import get_indexes as c_get_indexes

    # ===== Testing (0, 10, 100) ======
    print('===== Testing (0, 10, 100) =====')

    def run_py():
        d = py_get_indexes(0, 10, 100)

    def run_c():
        d = c_get_indexes(0, 10, 100)

    t1 = timeit.timeit(run_py)
    print('Py Time:', t1)
    t2 = timeit.timeit(run_c)
    print('C Time:', t2)
    assert t1 > t2

    # ===== Testing (5, 10, 12) ======
    print('===== Testing (5, 10, 12) =====')

    def run_py():
        d = py_get_indexes(5, 10, 12)

    def run_c():
        d = c_get_indexes(5, 10, 12)

    t1 = timeit.timeit(run_py)
    print('Py Time:', t1)
    t2 = timeit.timeit(run_c)
    print('C Time:', t2)
    assert t1 > t2

    # ===== Testing (0, 1000, 1000) ======
    print('===== Testing (0, 1000, 1000) =====')

    def run_py():
        d = py_get_indexes(0, 1000, 1000)

    def run_c():
        d = c_get_indexes(0, 1000, 1000)

    t1 = timeit.timeit(run_py)
    print('Py Time:', t1)
    t2 = timeit.timeit(run_c)
    print('C Time:', t2)
    assert t1 > t2

    # ===== Testing (500, 1000, 1000) ======
    print('===== Testing (500, 1000, 1000) =====')

    def run_py():
        d = py_get_indexes(500, 1000, 1000)

    def run_c():
        d = c_get_indexes(500, 1000, 1000)

    t1 = timeit.timeit(run_py)
    print('Py Time:', t1)
    t2 = timeit.timeit(run_c)
    print('C Time:', t2)
    assert t1 > t2

    # ===== Testing (700, 1000, 1000) ======
    print('===== Testing (700, 1000, 1000) =====')

    def run_py():
        d = py_get_indexes(700, 1000, 1000)

    def run_c():
        d = c_get_indexes(700, 1000, 1000)

    t1 = timeit.timeit(run_py)
    print('Py Time:', t1)
    t2 = timeit.timeit(run_c)
    print('C Time:', t2)
    assert t1 > t2


if __name__ == '__main__':
    test_get_indexes()
    time_get_indexes()

    print('All tests finished successfully!')
