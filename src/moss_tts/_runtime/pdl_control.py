"""Gluon grid dependency instructions, following Triton's CUDA GDC interface.

Waiting is required before consuming producer data. Triggering only permits
opportunistic scheduling; kernels must never depend on concurrent execution.
"""
from triton.experimental import gluon
from triton.experimental.gluon import language as gl


@gluon.jit
def wait():
    gl.inline_asm_elementwise('griddepcontrol.wait; // dummy $0', '=r', [],
                             dtype=gl.int32, is_pure=False, pack=1)


@gluon.jit
def trigger():
    gl.inline_asm_elementwise('griddepcontrol.launch_dependents; // dummy $0', '=r', [],
                             dtype=gl.int32, is_pure=False, pack=1)
