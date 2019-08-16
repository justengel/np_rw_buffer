//
// Python Ring Buffer: New Numpy array circular buffer
// https://docs.python.org/3/extending/newtypes.html
//
// Helpful Links:
//   Additional Extending Python Help - https://docs.python.org/2.5/ext/node23.html
//   parsing-arguments - https://docs.python.org/3.4/c-api/arg.html
//   Extra variables - https://docs.python.org/2/c-api/structures.html#METH_CLASS
//   Packaging - https://packaging.python.org/guides/packaging-binary-extensions/
//   Py27 compiler - http://aka.ms/vcpython27
//   Py34 compiler - www.microsoft.com/download/details.aspx?id=8279
//   compilers - https://wiki.python.org/moin/WindowsCompilers
//
// Numpy C API:
//   http://folk.uio.no/inf3330/scripting/doc/python/NumPy/Numeric/numpy-13.html
//   https://stackoverflow.com/questions/214549/how-to-create-a-numpy-record-array-from-c
//
#include <inttypes.h>
#include <Python.h>
#include <numpy/arrayobject.h>


/*
 * get_indexes
 *
 * Args:
 *     start (int): Start index
 *     length (int): Length of indexes
 *     maxsize (int): Maximum length of the array for wrap around.
 *
 * Returns:
 *     idxs (tuple): Tuple of indexes with wrap around support.
*/
static PyObject *
get_indexes(PyObject *self, PyObject* args, PyObject *kwds)
{
    // Argument variables
    int start;
    int length;
    int maxsize;

    npy_intp dims[1] = {0};
    PyObject *arr;
    int32_t *data;
    int i;
    int end;

    // Parse the arguments
    static char *kwlist[] = {"start", "length", "maxsize", NULL};
    if (! PyArg_ParseTupleAndKeywords(args, kwds, "III", kwlist, &start, &length, &maxsize))
        return NULL;

    if(length < 0){
        PyErr_SetString(PyExc_ValueError, "The given length must be greater than or equal to 0.");
        return NULL;
    }

    // ===== Check if slice will work =====
    end = start + length;
    if(end <= maxsize){
        return PySlice_New(PyLong_FromLong(start), PyLong_FromLong(end), PyLong_FromLong(1));
    }

    // ===== Build an array =====
    dims[0] = length;
    arr = (PyObject *) PyArray_SimpleNew(1, dims, NPY_INT32);
    if(arr == NULL){
        PyErr_SetString(PyExc_ValueError, "Cannot create the array with the size of the given length.");
        return NULL;
    }

    data = (int32_t *) PyArray_DATA(arr);
    for(i=0; i < length; i++){
        data[i] = (start + i) % maxsize;
    }

    return arr;
}


// Required build items
static PyMethodDef circular_indexes_module_methods[] = {
    {"get_indexes", get_indexes, METH_VARARGS | METH_KEYWORDS,
     "Return an array of indexes for the given range.\n"
     "\n"
     "Args:\n"
     "    start (int): Start index.\n"
     "    length (int): Length of indexes.\n"
     "    maxsize (int): Maximum length of the array for wrap around.\n"
     "\n"
     "Returns:\n"
     "    idxs (tuple): Tuple of indexes with wrap around support.\n"},


    {NULL}  /* Sentinel */
};


#if PY_MAJOR_VERSION >= 3
// Python 3.x

static struct PyModuleDef circular_indexes2module = {
    PyModuleDef_HEAD_INIT,
    "_circular_indexes",
    "Parse fields out from bytes and bits.",
    -1,
    circular_indexes_module_methods, NULL, NULL, NULL, NULL
};


PyMODINIT_FUNC
PyInit__circular_indexes(void)
{
    PyObject* m;

    m = PyModule_Create(&circular_indexes2module);
    if (m == NULL)
        return NULL;

    import_array();

    return m;
}

#else
// Python 2.7
#define INITERROR return


PyMODINIT_FUNC
init_circular_indexes(void)
{
    PyObject* m;

    m = Py_InitModule("_circular_indexes", circular_indexes_module_methods);
    if (m == NULL)
        return;

    import_array();
}

#endif
