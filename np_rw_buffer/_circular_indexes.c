#include <Python.h>


static PyObject *
get_indexes(PyObject* self, PyObject *args)
{
    long long start;
    long long length;
    long long maxsize;
    long long i;
    PyTupleObject *result;

    if (!PyArg_ParseTuple(args, "iii", &start, &length, &maxsize)){
        printf("err: %d", 1);
        return NULL;
    }

    result = PyTuple_New(length);
    if(result == NULL){
        printf("here: %d", 2);
        return NULL;
    }

    for(i=0; i < length; i++){
        // PyTuple_SetItem(result, i, (start %  maxsize) + i);
        PyTuple_SetItem(result, i, PyLong_FromLongLong((start %  maxsize) + i));
    }

    if (PyErr_Occurred()) {
        printf("error occurred: %d", 3);
        return NULL;
    } else {
        return result;
    }
}


// Required build items
static PyMethodDef _circular_indexes_module_methods[] = {
    {"get_indexes", get_indexes, METH_VARARGS,
     "Return a tuple of circular indexes from the given start, length, and maxsize arguments.\n"
     "\n"
     "Args:\n"
     "    start (int): Start index.\n"
     "    length (int): Number of indices to populate.\n"
     "    maxsize (int): Maximum size of the array. This is used for the wrap around to make the array circular.\n"
     "\n"
     "Returns:\n"
     "    results (tuple): Tuple of indices.\n"},

    {NULL}  /* Sentinel */
};


// Note: "_pysbin" is the compiled name. It will compile to "_pysbin.pyd"
#if PY_MAJOR_VERSION >= 3
// Python 3.x

static struct PyModuleDef _circular_indexes2module = {
    PyModuleDef_HEAD_INIT,
    "_circular_indexes",
    "Quickly grab circular indexes to make numpy circular arrays fast.",
    -1,
    _circular_indexes_module_methods, NULL, NULL, NULL, NULL
};


PyMODINIT_FUNC
PyInit__circular_indexes(void)
{
    PyObject* m;

    m = PyModule_Create(&_circular_indexes2module);
    if (m == NULL)
        return NULL;
    return m;
}

#else
// Python 2.7
#define INITERROR return


PyMODINIT_FUNC
init_circular_indexes(void)
{
    PyObject* m;

    m = Py_InitModule("_circular_indexes", _circular_indexes_module_methods);
    if (m == NULL)
        return;
}

#endif
