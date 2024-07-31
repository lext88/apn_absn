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


#include <ATen/ATen.h>
#include <torch/extension.h>
#include "ddf_add_faster_forward_cuda.h"

int DDFAddFasterForwardLauncher(
    const at::Tensor features, const at::Tensor channel_filter,
    const at::Tensor spatial_filter, const int kernel_size,
    const int dilation, const int stride,
    const int batch_size, const int channels,
    const int bottom_length, const int top_length,
    at::Tensor output) {
    // Implementation of the CUDA kernel should be updated to handle 1D sequences
    // instead of 2D images. This is a placeholder for the actual kernel call.
    // Make sure to implement or adjust your CUDA kernel for 1D sequence processing.
    // e.g., call an updated kernel function here
    return 1;
}

int ddf_add_faster_forward_cuda(
    at::Tensor features, at::Tensor channel_filter, at::Tensor spatial_filter,
    int kernel_size, int dilation, int stride, at::Tensor output) {
    CHECK_INPUT(features);
    CHECK_INPUT(channel_filter);
    CHECK_INPUT(spatial_filter);
    CHECK_INPUT(output);
    at::DeviceGuard guard(features.device());

    const int batch_size = features.size(0);
    const int channels = features.size(1);
    const int bottom_length = features.size(2); // Assuming 1D sequence length
    const int top_length = output.size(2); // Assuming 1D sequence length

    DDFAddFasterForwardLauncher(features, channel_filter, spatial_filter,
                                kernel_size, dilation, stride,
                                batch_size, channels,
                                bottom_length, top_length,
                                output);
    return 1;
}
