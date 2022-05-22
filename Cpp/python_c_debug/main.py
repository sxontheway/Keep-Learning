import ctypes, numpy as np
from numpy.ctypeslib import ndpointer


lib = ctypes.cdll.LoadLibrary("./myadd.so")
fun = lib.cfun
fun.restype = None
fun.argtypes = [
    ndpointer(ctypes.c_double, flags="C_CONTIGUOUS"),
    ctypes.c_size_t,
    ndpointer(ctypes.c_double, flags="C_CONTIGUOUS")
]

indata = np.ones((5, 6))
outdata = np.empty((5, 6))
fun(indata, indata.size, outdata)
print(indata)
print(indata.size)
print(outdata)