"""
These are old tests from an old library, but they should still work.
"""
import unittest

import numpy as np
from np_rw_buffer import RingBuffer, AudioFramingBuffer


class TestBuffer(unittest.TestCase):
    """Test the basic buffer."""

    def setUp(self):
        """Setup the tests."""
        self.size = 10
        self.channels = 2
        self.dtype = np.float32
        self.buffer = RingBuffer(self.size, self.channels, self.dtype)
    # end setUp

    def test_clear(self):
        """Test the clear method."""
        self.test_write()
        self.buffer.clear()

        self.assertEqual(0, self.buffer._start, "Clear failed!")
        self.assertEqual(0, self.buffer._end, "Clear failed!")
        self.assertEqual(0, len(self.buffer), "Clear failed!")
    # end test_clear

    def test_size(self):
        """Test the buffers maximum size."""
        self.assertEqual(self.size, self.buffer.maxsize, "Invalid maximum size")
        self.buffer.maxsize = 12
        self.assertEqual(12, self.buffer.maxsize, "Invalid maximum size")

        self.assertEqual(self.channels, self.buffer.columns, "Invalid channels")
        self.buffer.columns = 4
        self.assertEqual(4, self.buffer.columns, "Invalid channels")
    # end test_size

    def test_dtype(self):
        """Test the buffer data type."""
        self.assertEqual(self.dtype, self.buffer.dtype, "Invalid dtype")
        self.buffer.dtype = np.int16
        self.assertEqual(np.int16, self.buffer.dtype, "Invalid dtype")
    # end test_size

    def test_data(self):
        """Test set and get data."""
        data = np.array([(i, i) for i in range(200)])
        self.buffer.set_data(data)
        self.assertTrue(np.all(data == self.buffer.get_data()), "Invalid set/get data!")
    # end test_data

    def test_write(self):
        """Test the write method."""
        # Write half
        data = [tuple(i for _ in range(self.channels)) for i in range(int(self.size//2))]
        self.buffer.write(data)
        self.assertEqual(len(data), len(self.buffer), "Invalid write!")

        # Check full
        self.buffer.write(data)
        self.assertEqual((self.buffer.maxsize//2) * 2, len(self.buffer), "Invalid write!")

        # Check overflow
        try:
            self.buffer.write(data)
            self.assertEqual((self.buffer.maxsize//2) * 2, len(self.buffer), "Invalid write!")
        except OverflowError:
            pass
    # end test_write

    def test_read(self):
        """Test the read method."""
        data = np.array([tuple(i for _ in range(self.channels)) for i in range(self.size//2)],
                        dtype=self.dtype)
        self.buffer.write(data)
        self.assertEqual(len(data), len(self.buffer), "Invalid read!")
        self.assertTrue(np.all(data == self.buffer.read()), "Invalid read!")
        self.assertEqual(0, len(self.buffer), "Invalid read!")
    # end test_read

    def test_move_start(self):
        """Test moving the start pointer."""
        with self.assertRaises(ValueError):
            self.buffer.move_start(5)

        # Move backwards to increase size
        self.buffer.move_start(-self.size//2)
        self.assertEqual((self.size//2), len(self.buffer), "Invalid move_start negative!")

        # Move backwards to make the buffer full
        self.buffer.clear()
        self.buffer.move_start(-self.size)
        self.assertEqual(self.size, len(self.buffer), "Invalid move_start negative to full!")

        # Move forward to decrease size
        self.buffer.move_start(self.size//2)
        self.assertEqual((self.size//2), len(self.buffer), "Invalid move_start positive!")

        # Move forward to make empty
        self.buffer.clear()
        self.buffer.move_start(-self.size)
        self.buffer.move_start(self.size)
        self.assertEqual(0, len(self.buffer), "Invalid move_start positive to empty!")
    # end test_move_start

    def test_move_end(self):
        """Test moving the end pointer."""
        # Check overflow error
        self.buffer.move_end(self.size-2, False)
        with self.assertRaises(OverflowError):
            self.buffer.move_end(5)

        # Move backwards past start to increase size
        self.buffer.clear()
        self.buffer.move_end(-self.size//2)
        self.assertEqual((self.size//2), len(self.buffer), "Invalid move_end negative!")

        # Move backwards to make the buffer empty
        self.buffer.clear()
        self.buffer.move_end(-self.size)
        self.assertEqual(0, len(self.buffer), "Invalid move_end negative to empty!")

        # Move forward to increase size
        self.buffer.move_end(self.size//2)
        self.assertEqual((self.size//2), len(self.buffer), "Invalid move_end positive!")

        # Move forward to make full
        self.buffer.clear()
        self.buffer.move_end(-self.size)
        self.buffer.move_end(self.size)
        self.assertEqual(self.size, len(self.buffer), "Invalid move_end positive to full!")
        self.assertEqual(self.buffer._start, self.buffer._end, "Invalid move_end positive to full!")

        self.buffer.move_end(1, False)
        self.assertEqual(self.size, len(self.buffer), "Invalid move_end positive to full!")
        self.assertEqual(self.buffer._start, self.buffer._end, "Invalid move_end positive to full!")

        self.buffer.move_end(1, False)
        self.assertEqual(self.size, len(self.buffer), "Invalid move_end positive to full!")
        self.assertEqual(self.buffer._start, self.buffer._end, "Invalid move_end positive to full!")

        self.buffer.move_end(1, False)
        self.assertEqual(self.size, len(self.buffer), "Invalid move_end positive to full!")
        self.assertEqual(self.buffer._start, self.buffer._end, "Invalid move_end positive to full!")

        self.buffer.move_end(5, False)
        self.assertEqual(self.size, len(self.buffer), "Invalid move_end positive to full!")
        self.assertEqual(self.buffer._start, self.buffer._end, "Invalid move_end positive to full!")

        self.buffer.move_end(2, False)
        self.assertEqual(self.size, len(self.buffer), "Invalid move_end positive to full!")
        self.assertEqual(self.buffer._start, self.buffer._end, "Invalid move_end positive to full!")
    # end test_move_end

    def test_copy(self):
        """Test if the data is copied and reference change errors happen."""
        test = np.array([(i, i) for i in range(10)], dtype=self.dtype)
        self.buffer.write(test)

        # Test outer assignment
        test[0, 0] = 50
        self.assertNotEqual(50, self.buffer._data[0, 0], "Invalid write copy!")

        # Test outer assignment
        test[0, 0] = 0
        self.buffer._data[0, 0] = 50
        self.assertNotEqual(50, test[0, 0], "Invalid write copy!")

        # Test internal assignment
        self.buffer._data[0][0] = 50
        self.assertNotEqual(50, test[0, 0], "Invalid write copy!")
        self.assertNotEqual(50, test[0][0], "Invalid write copy!")

        # Test read assignment
        self.buffer._data[0][0] = 0
        data = self.buffer.read()
        data[0][0] = 50
        self.assertNotEqual(50, self.buffer._data[0, 0], "Invalid read copy!")
        self.assertNotEqual(50, self.buffer._data[0][0], "Invalid read copy!")

        # Test read assignment
        data[0][0] = 0
        self.buffer._data[0, 0] = 50
        self.assertNotEqual(50, data[0][0], "Invalid read copy!")


        # Test get_data assignment
        self.buffer.write(data)
        data = self.buffer.get_data()
        data[0][0] = 50
        self.assertNotEqual(50, self.buffer._data[0, 0], "Invalid get_data copy!")
        self.assertNotEqual(50, self.buffer._data[0][0], "Invalid get_data copy!")

        # Test get_data assignment
        data[0][0] = 0
        self.buffer._data[0, 0] = 50
        self.assertNotEqual(50, data[0][0], "Invalid get_data copy!")

        # Test nulls
        d = self.buffer._data[0:0]
        self.buffer.write(data, error=False)
        self.assertEqual(0, len(d), "Invalid null data copy!")
    # end test_copy
# end class TestBuffer


class TestAudioFramingBuffer(TestBuffer):
    def setUp(self):
        """Setup the tests."""
        self.size = 10
        self.seconds = 2
        self.sample_rate = np.ceil(self.size / self.seconds)
        self.channels = 2
        self.dtype = np.float32
        self.buffer = AudioFramingBuffer(self.sample_rate, self.channels, seconds=self.seconds, dtype=self.dtype)
    # end setUp

    def test_write(self):
        """Test the write method."""
        super().test_write()

        data = [tuple(i for _ in range(self.channels)) for i in range(self.size//2)]

        # Check overflow
        try:
            self.buffer.write(data)
            self.assertEqual((self.buffer.maxsize//2) * 2, len(self.buffer), "Invalid write!")
        except OverflowError:
            # Check the Audio buffer for a fixed len to maxsize, but the underlying _length growing
            self.buffer.write(data, False)
            self.assertEqual((self.buffer.maxsize//2) * 2, len(self.buffer), "Invalid write!")
            self.assertEqual(self.buffer.maxsize + len(data), self.buffer._length, "Invalid write!")
    # end test_write

    def test_read(self):
        """Test the read method."""
        super().test_read()

        self.assertEqual(0, len(self.buffer), "Invalid read!")
    # end test_read

    def test_read_delay(self):
        """Test that the buffer read returns zeros until the delay."""
        self.buffer.set_sample_rate(self.size)
        self.buffer.buffer_delay = 1

    def test_move_start(self):
        """Test moving the start pointer."""
        return
        with self.assertRaises(ValueError):
            self.buffer.move_start(5)

        # Move backwards to increase size
        self.buffer.move_start(-self.size//2)
        self.assertEqual((self.size//2), len(self.buffer), "Invalid move_start negative!")

        # Move backwards to make the buffer full
        self.buffer.clear()
        self.buffer.move_start(-self.size)
        self.assertEqual(self.size, len(self.buffer), "Invalid move_start negative to full!")

        # Move forward to decrease size
        self.buffer.move_start(self.size//2)
        self.assertEqual((self.size//2), len(self.buffer), "Invalid move_start positive!")

        # Move forward to make empty
        self.buffer.clear()
        self.buffer.move_start(-self.size)
        self.buffer.move_start(self.size)
        self.assertEqual(0, len(self.buffer), "Invalid move_start positive to empty!")
    # end test_move_start

    def test_move_end(self):
        """Test moving the end pointer."""
        return
        # Check overflow error
        self.buffer.move_end(self.size-2, False)
        with self.assertRaises(OverflowError):
            self.buffer.move_end(5)

        # Move backwards past start to increase size
        self.buffer.clear()
        self.buffer.move_end(-self.size//2)
        self.assertEqual((self.size//2), len(self.buffer), "Invalid move_end negative!")

        # Move backwards to make the buffer empty
        self.buffer.clear()
        self.buffer.move_end(-self.size)
        self.assertEqual(0, len(self.buffer), "Invalid move_end negative to empty!")

        # Move forward to increase size
        self.buffer.move_end(self.size//2)
        self.assertEqual((self.size//2), len(self.buffer), "Invalid move_end positive!")

        # Move forward to make full
        self.buffer.clear()
        self.buffer.move_end(-self.size)
        self.buffer.move_end(self.size)
        self.assertEqual(self.size, len(self.buffer), "Invalid move_end positive to full!")
        self.assertEqual(self.buffer._start, self.buffer._end, "Invalid move_end positive to full!")

        self.buffer.move_end(1, False)
        self.assertEqual(self.size, len(self.buffer), "Invalid move_end positive to full!")
        self.assertEqual(self.buffer._start, self.buffer._end, "Invalid move_end positive to full!")

        self.buffer.move_end(1, False)
        self.assertEqual(self.size, len(self.buffer), "Invalid move_end positive to full!")
        self.assertEqual(self.buffer._start, self.buffer._end, "Invalid move_end positive to full!")

        self.buffer.move_end(1, False)
        self.assertEqual(self.size, len(self.buffer), "Invalid move_end positive to full!")
        self.assertEqual(self.buffer._start, self.buffer._end, "Invalid move_end positive to full!")

        self.buffer.move_end(5, False)
        self.assertEqual(self.size, len(self.buffer), "Invalid move_end positive to full!")
        self.assertEqual(self.buffer._start, self.buffer._end, "Invalid move_end positive to full!")

        self.buffer.move_end(2, False)
        self.assertEqual(self.size, len(self.buffer), "Invalid move_end positive to full!")
        self.assertEqual(self.buffer._start, self.buffer._end, "Invalid move_end positive to full!")
    # end test_move_end

    def test_copy(self):
        """Test if the data is copied and reference change errors happen."""
        return
        test = np.array([(i, i) for i in range(10)], dtype=self.dtype)
        self.buffer.write(test)

        # Test outer assignment
        test[0, 0] = 50
        self.assertNotEqual(50, self.buffer._data[0, 0], "Invalid write copy!")

        # Test outer assignment
        test[0, 0] = 0
        self.buffer._data[0, 0] = 50
        self.assertNotEqual(50, test[0, 0], "Invalid write copy!")

        # Test internal assignment
        self.buffer._data[0][0] = 50
        self.assertNotEqual(50, test[0, 0], "Invalid write copy!")
        self.assertNotEqual(50, test[0][0], "Invalid write copy!")

        # Test read assignment
        self.buffer._data[0][0] = 0
        data = self.buffer.read()
        data[0][0] = 50
        self.assertNotEqual(50, self.buffer._data[0, 0], "Invalid read copy!")
        self.assertNotEqual(50, self.buffer._data[0][0], "Invalid read copy!")

        # Test read assignment
        data[0][0] = 0
        self.buffer._data[0, 0] = 50
        self.assertNotEqual(50, data[0][0], "Invalid read copy!")

        # Test get_data assignment
        self.buffer.write(data)
        data = self.buffer.get_data()
        data[0][0] = 50
        self.assertNotEqual(50, self.buffer._data[0, 0], "Invalid get_data copy!")
        self.assertNotEqual(50, self.buffer._data[0][0], "Invalid get_data copy!")

        # Test get_data assignment
        data[0][0] = 0
        self.buffer._data[0, 0] = 50
        self.assertNotEqual(50, data[0][0], "Invalid get_data copy!")

        # Test nulls
        d = self.buffer._data[0:0]
        self.buffer.write(data)
        self.assertEqual(0, len(d), "Invalid null data copy!")
    # end test_copy
# end class TestAudioFramingBuffer


if __name__ == "__main__":
    unittest.main()

