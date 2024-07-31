#include <ATen/ATen.h>
#include <torch/extension.h>

#include <cmath>
#include <vector>

#ifdef WITH_CUDA
    // CUDA function declaration for handling text data
    int ddf_add_faster_forward_cuda(
        at::Tensor features, at::Tensor channel_filter, at::Tensor spatial_filter,
        int kernel_size, int dilation, int stride, at::Tensor output);
#endif

int ddf_add_faster_forward(
    at::Tensor features, at::Tensor channel_filter, at::Tensor spatial_filter,
    int kernel_size, int dilation, int stride, at::Tensor output) {
    // Check if the tensors are on CUDA
    if (features.device().is_cuda()) {
        #ifdef WITH_CUDA
            return ddf_add_faster_forward_cuda(
                features, channel_filter, spatial_filter,
                kernel_size, dilation, stride, output);
        #else
            AT_ERROR("DDF operation is not compiled with GPU support");
        #endif
    }
    AT_ERROR("DDF operation is not implemented on CPU");
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &ddf_add_faster_forward, "DDF add faster forward for text data");
}
