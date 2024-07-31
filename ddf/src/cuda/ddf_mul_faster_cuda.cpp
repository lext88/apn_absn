#include <ATen/ATen.h>
#include <torch/extension.h>

#include <cmath>
#include <vector>

int DDFMulFasterForwardLauncher(
    const at::Tensor features, const at::Tensor channel_filter,
    const at::Tensor spatial_filter, const int kernel_size,
    const int dilation, const int stride,
    const int batch_size, const int sequence_length,
    const int embedding_dim, const int output_length,
    at::Tensor output);

#define CHECK_CUDA(x) TORCH_CHECK(x.device().is_cuda(), #x, " must be a CUDA tensor ")
#define CHECK_CONTIGUOUS(x) \
    TORCH_CHECK(x.is_contiguous(), #x, " must be contiguous ")
#define CHECK_INPUT(x) \
    CHECK_CUDA(x);       \
    CHECK_CONTIGUOUS(x)

int ddf_mul_faster_forward_cuda(
    at::Tensor features, at::Tensor channel_filter, at::Tensor spatial_filter,
    int kernel_size, int dilation, int stride, at::Tensor output) {
    CHECK_INPUT(features);
    CHECK_INPUT(channel_filter);
    CHECK_INPUT(spatial_filter);
    CHECK_INPUT(output);
    at::DeviceGuard guard(features.device());

    const int batch_size = features.size(0);
    const int sequence_length = features.size(1);
    const int embedding_dim = features.size(2);
    const int output_length = output.size(1);

    DDFMulFasterForwardLauncher(features, channel_filter, spatial_filter,
                                kernel_size, dilation, stride,
                                batch_size, sequence_length,
                                embedding_dim, output_length,
                                output);
    return 1;
}
