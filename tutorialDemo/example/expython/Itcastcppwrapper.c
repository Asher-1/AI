#include "Python.h"
#include <stdlib.h>
#include <string.h>
#include "Itcastcpp.h"

static PyObject *Itcastcpp_fac(PyObject *self, PyObject *args)
{
    int num;

    if (!PyArg_ParseTuple(args, "i", &num))
        return NULL;

    return (PyObject *)Py_BuildValue("i", fac(num));
}

static PyObject *Itcastcpp_doppel(PyObject *self, PyObject *args)
{
    char *src;
    char *mstr;
    PyObject *retval;

    if (!PyArg_ParseTuple(args, "s", &src))
        return NULL;

    mstr = malloc(strlen(src) + 1);
    strcpy(mstr, src);
    reverse(mstr);
    retval = (PyObject *)Py_BuildValue("ss", src, mstr);
    free(mstr);

    return retval;
}

static PyObject *Itcastcpp_test(PyObject *self, PyObject *args)
{
    test();

    return (PyObject *)Py_BuildValue("");
}

static PyMethodDef ItcastcppMethods[] = {
    {"fac", Itcastcpp_fac, METH_VARARGS},
    {"doppel", Itcastcpp_doppel, METH_VARARGS},
    {"test", Itcastcpp_test, METH_VARARGS},
    {NULL, NULL},
};

void initItcastcpp(void)
{
    Py_InitModule("Itcastcpp", ItcastcppMethods);
}
