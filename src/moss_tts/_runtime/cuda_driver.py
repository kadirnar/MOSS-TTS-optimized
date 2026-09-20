"""CUDA driver launch structures for the bundled SM90 kernels."""
import ctypes as C

class AttributeValue(C.Union):
    _fields_ = [('pad', C.c_char * 64), ('serialization', C.c_int), ('alignment', C.c_uint64)]


class Attribute(C.Structure):
    _fields_ = [('id', C.c_int), ('pad', C.c_char * 4), ('value', AttributeValue)]


class LaunchConfig(C.Structure):
    _fields_ = [(name, C.c_uint) for name in ('gx','gy','gz','bx','by','bz','shared')]+[
        ('stream', C.c_void_p), ('attrs', C.POINTER(Attribute)), ('count', C.c_uint)]


def check(result, operation):
    if result: raise RuntimeError(f'{operation}: CUDA driver error {result}')
