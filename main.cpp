#include <pybind11/pybind11.h>
#include <torch/torch.h>
#include <ATen/ATen.h>
// #include "torchac_kernel.cuh"
#ifdef __HIP_PLATFORM_HCC__
#include "torchac_kernel_hip.cuh"
#else
#include "torchac_kernel.cuh"
#endif

namespace py = pybind11;

PYBIND11_MODULE(torchac_cuda, m) {
    m.def("encode_fast_new", &encode_cuda_new);
    m.def("decode_fast_new", &decode_cuda_new);
    m.def("decode_fast_prefsum", &decode_cuda_prefsum);
    m.def("calculate_cdf", &calculate_cdf);
}
