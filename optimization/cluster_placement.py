"""Experimental launch-only cluster placement and shared/L1 carveout policies.

The selected cubin and its arithmetic are unchanged. Each variant owns fresh
driver modules; launch attributes do not alter another model or CUDA context.
"""
import ctypes as C
import json
from pathlib import Path
from types import SimpleNamespace
import torch
from .qkv_cluster_binary import Binary
from .qkv_cluster_prepare import launch
from .ptx_resources import Attribute,LaunchConfig,check


class PlacementBinary(Binary):
    def __init__(self,folder,record,*,policy=1,carveout=None):
        super().__init__(folder,record)
        if policy not in (0,1,2) or carveout not in (None,0,25,50,100):raise ValueError('Unsupported placement policy')
        self.policy=policy;self.carveout=carveout;self.occupancy=None
        self.driver=self.lib
        self.driver.cuOccupancyMaxActiveClusters.argtypes=[C.POINTER(C.c_int),C.c_void_p,C.POINTER(LaunchConfig)]
        self.lib=SimpleNamespace(cuLaunchKernelEx=self._launch)

    def _launch(self,config_ptr,function,pointers,extra):
        original=C.cast(config_ptr,C.POINTER(LaunchConfig)).contents
        assert original.count==3 and original.attrs[2].id==5
        config=LaunchConfig.from_buffer_copy(original)
        attrs=(Attribute*4)()
        C.memmove(attrs,original.attrs,3*C.sizeof(Attribute))
        attrs[2].value.serialization=self.policy
        if self.carveout is not None:
            attrs[3].id=14;attrs[3].value.serialization=self.carveout
            config.count=4
        config.attrs=attrs
        if self.occupancy is None:
            value=C.c_int()
            check(self.driver.cuOccupancyMaxActiveClusters(C.byref(value),function,C.byref(config)),'cluster occupancy query')
            self.occupancy={'max_active_clusters':value.value,'policy':self.policy,'carveout_percent':self.carveout,
                'module_handle':self.module.value,'function_handle':self.function.value}
        return self.driver.cuLaunchKernelEx(C.byref(config),function,pointers,extra)


def configured(folder,*,policy=1,carveout=None):
    folder=Path(folder);manifest=json.loads((folder/'manifest.json').read_text())
    if manifest['format']!='qkv_cluster_cubin_v1' or not manifest['complete'] or torch.cuda.get_device_capability()!=(9,0):
        raise ValueError('Complete SM90 bundle required')
    torch.empty(0,device='cuda')
    name='c8_t2_exact';options=manifest['configs'][name]
    kernels={n:PlacementBinary(folder,r,policy=policy,carveout=carveout) for n,r in manifest['binaries'].items() if n.startswith(name+'_add')}
    def fn(*args,**kwargs):
        if any(kwargs.get(k)!=v for k,v in options.items()):raise ValueError('Different kernel configuration')
        key=f'{name}_add{int(args[1] is not None)}_debug{int(kwargs.get("debug",False))}'
        return launch(*args,**kwargs,compiled=kernels[key])
    fn.kernels=kernels
    return options,fn
