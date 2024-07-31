#include <ATen/ATen.h>
#include <torch/extension.h>

#include <cmath>
#include <vector>

int DDFAddFasterForwardLauncher(
    const at::Tensor features, const at::Tensor channel_filter,
    const at::Tensor spatial_filter, const int kernel_size,
    const int dilation, const int stride,
    const int batch_size, const int channels,
    const int bottom_length, const int top_length,
    at::Tensor output);

#define CHECK_CUDA(x) TORCH_CHECK(x.device().is_cuda(), #x, " must be a CUDA tensor ")
#define CHECK_CONTIGUOUS(x) \
    TORCH_CHECK(x.is_contiguous(), #x, " must be contiguous ")
#define CHECK_INPUT(x) \
    CHECK_CUDA(x);       \
    CHECK_CONTIGUOUS(x)

int ddf_add_faster_forward_cuda(
    at::Tensor features, at::Tensor channel_filter, at::Tensor spatial_filter,
    int kernel_size, int dilation, int stride, at::Tensor output);
