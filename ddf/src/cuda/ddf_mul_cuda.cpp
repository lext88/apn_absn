#include <ATen/ATen.h>
#include <torch/extension.h>

#include <cmath>
#include <vector>

// Function declarations
int DDFMulForwardLauncher(
    const at::Tensor features, const at::Tensor channel_filter,
    const at::Tensor spatial_filter, const int kernel_size,
    const int dilation, const int stride,
    const int batch_size, const int channels,
    const int bottom_length, const int top_length,
    at::Tensor output);

int DDFMulBackwardLauncher(
    const at::Tensor top_grad, const at::Tensor features,
    const at::Tensor channel_filter, const at::Tensor spatial_filter,
    const int kernel_size, const int dilation, const int stride,
    const int batch_size, const int channels,
    const int top_length, const int bottom_length,
    at::Tensor rtop_grad, at::Tensor rbottom_grad,
    at::Tensor rspatial_filter_grad, at::Tensor bottom_grad,
    at::Tensor channel_filter_grad, at::Tensor spatial_filter_grad);

// Utility macros for input validation
#define CHECK_CUDA(x) TORCH_CHECK(x.device().is_cuda(), #x, " must be a CUDA tensor ")
#define CHECK_CONTIGUOUS(x) \
    TORCH_CHECK(x.is_contiguous(), #x, " must be contiguous ")
#define CHECK_INPUT(x) \
    CHECK_CUDA(x);       \
    CHECK_CONTIGUOUS(x)

// Forward pass function
int ddf_mul_forward_cuda(
    at::Tensor features, at::Tensor channel_filter, at::Tensor spatial_filter,
    int kernel_size, int dilation, int stride, at::Tensor output) {
    CHECK_INPUT(features);
    CHECK_INPUT(channel_filter);
    CHECK_INPUT(spatial_filter);
    CHECK_INPUT(output);
    at::DeviceGuard guard(features.device());

    const int batch_size = features.size(0);
    const int channels = features.size(1);
    const int bottom_length = features.size(2);
    const int top_length = output.size(2);

    DDFMulForwardLauncher(features, channel_filter, spatial_filter,
                          kernel_size, dilation, stride,
                          batch_size, channels,
                          bottom_length, top_length,
                          output);
    return 1;
}

// Backward pass function
int ddf_mul_backward_cuda(
    at::Tensor top_grad, at::Tensor features,
    at::Tensor channel_filter, at::Tensor spatial_filter,
    int kernel_size, int dilation, int stride,
    at::Tensor rtop_grad, at::Tensor rbottom_grad,
    at::Tensor rspatial_filter_grad, at::Tensor bottom_grad,
    at::Tensor channel_filter_grad, at::Tensor spatial_filter_grad) {
    CHECK_INPUT(top_grad);
    CHECK_INPUT(features);
    CHECK_INPUT(channel_filter);
    CHECK_INPUT(spatial_filter);
    CHECK_INPUT(rtop_grad);
    CHECK_INPUT(rbottom_grad);
    CHECK_INPUT(rspatial_filter_grad);
    CHECK_INPUT(bottom_grad);
    CHECK_INPUT(channel_filter_grad);
    CHECK_INPUT(spatial_filter_grad);
    at::DeviceGuard guard(top_grad.device());

    const int batch_size = features.size(0);
    const int channels = features.size(1);
    const int bottom_length = features.size(2);
    const int top_length = top_grad.size(2);

    rtop_grad.resize_({batch_size, int(top_length / stride), channels});
    rbottom_grad.resize_({batch_size, bottom_length, channels});
    rspatial_filter_grad.resize_({batch_size, int(top_length / stride), kernel_size});

    DDFMulBackwardLauncher(top_grad, features, channel_filter, spatial_filter,
                           kernel_size, dilation, stride, batch_size,
                           channels, top_length, bottom_length,
                           rtop_grad, rbottom_grad, rspatial_filter_grad,
                           bottom_grad, channel_filter_grad, spatial_filter_grad);
    return 1;
}
