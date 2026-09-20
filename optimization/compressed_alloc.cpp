// Experimental CUDA VMM allocation owner, exposed through legacy DLPack.
// No model arithmetic or global allocator replacement is performed here.
#define PY_SSIZE_T_CLEAN
#include <Python.h>
#include <cuda.h>
#include <ATen/dlpack.h>
#include <atomic>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

static std::atomic<size_t> live_bytes{0}, live_allocations{0}, cleanup_errors{0};

static void check(CUresult result, const char* call) {
    if (result == CUDA_SUCCESS) return;
    const char* message = nullptr;
    cuGetErrorString(result, &message);
    throw std::runtime_error(std::string(call) + ": " + (message ? message : "CUDA error"));
}

struct Allocation {
    DLManagedTensor managed{};
    std::vector<int64_t> shape;
    CUcontext context = nullptr;
    CUdeviceptr pointer = 0;
    CUmemGenericAllocationHandle handle = 0;
    size_t size = 0;
    bool mapped = false, counted = false;

    ~Allocation() {
        // External memory is not tracked by Torch's caching allocator. Wait on
        // its owning context at final storage destruction, including all streams.
        // Graph users must keep their tensor owners alive until graph destruction.
        if (context && (pointer || handle)) {
            if (cuCtxPushCurrent(context) == CUDA_SUCCESS) {
                auto track = [](CUresult r) { if (r != CUDA_SUCCESS) ++cleanup_errors; };
                track(cuCtxSynchronize());
                if (mapped) track(cuMemUnmap(pointer, size));
                if (handle) track(cuMemRelease(handle));
                if (pointer) track(cuMemAddressFree(pointer, size));
                CUcontext previous;
                track(cuCtxPopCurrent(&previous));
            } else {
                ++cleanup_errors;
            }
        }
        if (counted) { live_bytes -= size; --live_allocations; }
    }
};

static void tensor_delete(DLManagedTensor* tensor) {
    delete static_cast<Allocation*>(tensor->manager_ctx);
}

static void capsule_delete(PyObject* capsule) {
    if (PyCapsule_IsValid(capsule, "dltensor")) {
        auto* tensor = static_cast<DLManagedTensor*>(PyCapsule_GetPointer(capsule, "dltensor"));
        tensor->deleter(tensor);
    }
}

static PyObject* allocate(PyObject*, PyObject* args) {
    PyObject* dimensions;
    int code, bits, device, compressed;
    if (!PyArg_ParseTuple(args, "Oiiii", &dimensions, &code, &bits, &device, &compressed)) return nullptr;
    auto* a = new Allocation;
    PyObject* sequence = nullptr;
    try {
        if (!((code == kDLInt && (bits == 8 || bits == 32 || bits == 64)) ||
              (code == kDLUInt && bits == 8) ||
              (code == kDLFloat && (bits == 16 || bits == 32 || bits == 64)) ||
              (code == kDLBfloat && bits == 16)))
            throw std::runtime_error("Unsupported DLPack scalar type");
        sequence = PySequence_Fast(dimensions, "shape must be a sequence");
        if (!sequence) { delete a; return nullptr; }
        size_t bytes = bits / 8;
        const Py_ssize_t ndim = PySequence_Fast_GET_SIZE(sequence);
        if (ndim > 32) throw std::runtime_error("At most 32 dimensions supported");
        for (Py_ssize_t i = 0; i < ndim; ++i) {
            auto dim = PyLong_AsLongLong(PySequence_Fast_GET_ITEM(sequence, i));
            if (PyErr_Occurred()) { Py_DECREF(sequence); delete a; return nullptr; }
            if (dim <= 0 || static_cast<uint64_t>(dim) > std::numeric_limits<size_t>::max() / bytes)
                throw std::runtime_error("Positive, non-overflowing dimensions required");
            a->shape.push_back(dim);
            bytes *= dim;
        }
        Py_CLEAR(sequence);
        check(cuCtxGetCurrent(&a->context), "cuCtxGetCurrent");
        if (!a->context) throw std::runtime_error("Initialize Torch CUDA context first");
        CUdevice current;
        check(cuCtxGetDevice(&current), "cuCtxGetDevice");
        if (current != device) throw std::runtime_error("Current CUDA context/device mismatch");
        int supported;
        check(cuDeviceGetAttribute(&supported, CU_DEVICE_ATTRIBUTE_VIRTUAL_ADDRESS_MANAGEMENT_SUPPORTED, device), "VMM support");
        if (!supported) throw std::runtime_error("Device does not support VMM");
        check(cuDeviceGetAttribute(&supported, CU_DEVICE_ATTRIBUTE_GENERIC_COMPRESSION_SUPPORTED, device), "Compression support");
        if (compressed && !supported) throw std::runtime_error("Device does not support generic compression");
        CUmemAllocationProp requested{};
        requested.type = CU_MEM_ALLOCATION_TYPE_PINNED;
        requested.location = {CU_MEM_LOCATION_TYPE_DEVICE, device};
        requested.allocFlags.compressionType = compressed ? CU_MEM_ALLOCATION_COMP_GENERIC : CU_MEM_ALLOCATION_COMP_NONE;
        size_t granularity;
        check(cuMemGetAllocationGranularity(&granularity, &requested, CU_MEM_ALLOC_GRANULARITY_MINIMUM), "cuMemGetAllocationGranularity");
        if (bytes > std::numeric_limits<size_t>::max() - granularity) throw std::runtime_error("Rounded allocation overflow");
        a->size = ((bytes + granularity - 1) / granularity) * granularity;
        check(cuMemCreate(&a->handle, a->size, &requested, 0), "cuMemCreate");
        CUmemAllocationProp actual{};
        check(cuMemGetAllocationPropertiesFromHandle(&actual, a->handle), "cuMemGetAllocationPropertiesFromHandle");
        // No silent fallback: every purported compressed allocation is checked.
        if (actual.allocFlags.compressionType != requested.allocFlags.compressionType)
            throw std::runtime_error("CUDA did not grant the requested compression property");
        check(cuMemAddressReserve(&a->pointer, a->size, granularity, 0, 0), "cuMemAddressReserve");
        check(cuMemMap(a->pointer, a->size, 0, a->handle, 0), "cuMemMap");
        a->mapped = true;
        CUmemAccessDesc access{};
        access.location = requested.location;
        access.flags = CU_MEM_ACCESS_FLAGS_PROT_READWRITE;
        check(cuMemSetAccess(a->pointer, a->size, &access, 1), "cuMemSetAccess");
        a->managed.dl_tensor = {reinterpret_cast<void*>(a->pointer), {kDLCUDA, device},
            static_cast<int32_t>(ndim), {static_cast<uint8_t>(code), static_cast<uint8_t>(bits), 1},
            a->shape.data(), nullptr, 0};
        a->managed.manager_ctx = a;
        a->managed.deleter = tensor_delete;
        a->counted = true;
        live_bytes += a->size; ++live_allocations;
        PyObject* metadata = Py_BuildValue("{s:K,s:K,s:K,s:i,s:i}",
            "logical_bytes", static_cast<unsigned long long>(bytes),
            "allocated_bytes", static_cast<unsigned long long>(a->size),
            "granularity", static_cast<unsigned long long>(granularity),
            "compression_type", static_cast<int>(actual.allocFlags.compressionType),
            "device", device);
        if (!metadata) { delete a; return nullptr; }
        PyObject* capsule = PyCapsule_New(&a->managed, "dltensor", capsule_delete);
        if (!capsule) { Py_DECREF(metadata); delete a; return nullptr; }
        return Py_BuildValue("NN", capsule, metadata);
    } catch (const std::exception& error) {
        Py_XDECREF(sequence);
        delete a;
        PyErr_SetString(PyExc_RuntimeError, error.what());
        return nullptr;
    }
}

static PyObject* counters(PyObject*, PyObject*) {
    return Py_BuildValue("{s:K,s:K,s:K}",
        "live_bytes", static_cast<unsigned long long>(live_bytes.load()),
        "live_allocations", static_cast<unsigned long long>(live_allocations.load()),
        "cleanup_errors", static_cast<unsigned long long>(cleanup_errors.load()));
}

static PyMethodDef methods[] = {
    {"allocate", allocate, METH_VARARGS, "Create a verified VMM allocation and owning DLPack capsule."},
    {"counters", counters, METH_NOARGS, "External allocations, excluded from Torch allocator statistics."},
    {nullptr, nullptr, 0, nullptr}
};
static PyModuleDef module = {PyModuleDef_HEAD_INIT, "_moss_compressed_alloc", nullptr, -1, methods};
PyMODINIT_FUNC PyInit__moss_compressed_alloc() { return PyModule_Create(&module); }
