#include <ATen/ATen.h>
#include <torch/extension.h>

#include <cmath>
#include <vector>

#define THREADS_PER_BLOCK 1024
#define DATA_TILE 16
#define CHANNEL_THREADS 4
#define CHANNEL_BLOCKS 8
#define MAX_KS 4
#define FULL_MASK 0xffffffff

inline int divideUP(const int x, const int y) { return (((x) + (y)-1) / (y)); }

__device__ inline int Loc2Index(const int n, const int c, const int i,
                                const int channel_num, const int length) {
    int index = i + (c + n * channel_num) * length;
    return index;
}

template <typename scalar_t>
__device__ inline scalar_t min(scalar_t a, scalar_t b) {
    return a < b ? a : b;
}

template <typename scalar_t>
__device__ inline scalar_t max(scalar_t a, scalar_t b) {
    return a > b ? a : b;
}

template <typename scalar_t>
__global__ void DDFForward(const scalar_t *__restrict__ bottom_data,
                           const scalar_t *__restrict__ bottom_channel_filter,
                           const scalar_t *__restrict__ bottom_spatial_filter,
                           const int kernel_size, const int dilation,
                           const int stride, const int padding,
                           const int batch_size, const int channels,
                           const int bottom_length, const int top_length,
                           scalar_t *__restrict__ top_data) {
    __shared__ scalar_t shared_spatial_filter[DATA_TILE * MAX_KS];
    __shared__ scalar_t shared_channel_filter[CHANNEL_THREADS * MAX_KS];
    __shared__ scalar_t shared_data[CHANNEL_THREADS * DATA_TILE];

    const int b = blockIdx.x;
    const int cb_id = blockIdx.y;
    bool valid_index = false;

    int top_i = -999999;

    if ((threadIdx.x - padding) % stride == 0) {
        top_i = (threadIdx.x - padding) / stride;
    }

    if (top_i >= 0 && top_i < top_length) {
        valid_index = true;
    }

    const int start_i = top_i * stride;
    const int end_i = start_i + 2 * padding + 1;

    const int bottom_i = blockIdx.x * DATA_TILE * stride - padding + threadIdx.x;

    if (valid_index) {
        if (top_i < top_length) {
            for (int i = threadIdx.z; i < kernel_size; i += CHANNEL_THREADS) {
                int spatial_filter_id = Loc2Index(b, i, top_i, kernel_size, top_length);
                shared_spatial_filter[top_i * kernel_size + i] =
                    bottom_spatial_filter[spatial_filter_id];
            }
        } else {
            for (int i = threadIdx.z; i < kernel_size; i += CHANNEL_THREADS) {
                shared_spatial_filter[top_i * kernel_size + i] = 0;
            }
        }
    }
    __syncthreads();

    #pragma unroll
    for (int c = cb_id * CHANNEL_THREADS + threadIdx.z; c < channels; c += CHANNEL_BLOCKS * CHANNEL_THREADS) {
        __syncthreads();

        if (threadIdx.x < kernel_size) {
            int channel_filter_id = ((b * channels + c) * kernel_size + threadIdx.x);
            shared_channel_filter[threadIdx.z * kernel_size + threadIdx.x] =
                bottom_channel_filter[channel_filter_id];
        }

        if (bottom_i >= 0 && bottom_i < bottom_length) {
            int id = Loc2Index(b, c, bottom_i, channels, bottom_length);
            shared_data[threadIdx.z * DATA_TILE + threadIdx.x] = bottom_data[id];
        } else {
            shared_data[threadIdx.z * DATA_TILE + threadIdx.x] = 0;
        }
        __syncthreads();

        if (valid_index && top_i < top_length) {
            scalar_t output_val = 0;
            scalar_t lost = 0;
            scalar_t t = 0;
            scalar_t input = 0;

            #pragma unroll
            for (int i = start_i; i < end_i; i += dilation) {
                int kernel_i = (i - start_i) / dilation;

                input = shared_data[threadIdx.z * DATA_TILE + i] *
                    (shared_spatial_filter[top_i * kernel_size + kernel_i] +
                    shared_channel_filter[threadIdx.z * kernel_size + kernel_i]);

                t = output_val + input;
                lost += fabs(output_val) >= fabs(input) ? (output_val - t) + input
                                                        : (input - t) + output_val;
                output_val = t;
            }

            int top_id = Loc2Index(b, c, top_i, channels, top_length);
            top_data[top_id] = output_val + lost;
        }
    }
}

int DDFAddFasterForwardLauncher(const at::Tensor features, const at::Tensor channel_filter,
                                const at::Tensor spatial_filter, const int kernel_size,
                                const int dilation, const int stride, const int padding,
                                const int batch_size, const int channels,
                                const int bottom_length, const int top_length,
                                at::Tensor output) {
    cudaStream_t stream = at::cuda::getCurrentCUDAStream();
    const int blocks_x = batch_size;
    const int blocks_y = divideUP(channels, CHANNEL_BLOCKS * CHANNEL_THREADS);
    dim3 grid(blocks_x, blocks_y);
    dim3 block(DATA_TILE, 1, CHANNEL_THREADS);

    AT_DISPATCH_FLOATING_TYPES_AND_HALF(
        features.type(), "DDFForward", ([&] {
            const scalar_t *bottom_data = features.data<scalar_t>();
            const scalar_t *bottom_channel_filter = channel_filter.data<scalar_t>();
            const scalar_t *bottom_spatial_filter = spatial_filter.data<scalar_t>();
            scalar_t *top_data = output.data<scalar_t>();
            DDFForward<scalar_t><<<grid, block, 0, stream>>>(
                bottom_data, bottom_channel_filter, bottom_spatial_filter,
                kernel_size, dilation, stride, padding, batch_size,
                channels, bottom_length, top_length, top_data);
    }));
    cudaError_t err = cudaGetLastError();
    if (cudaSuccess != err) {
        fprintf(stderr, "cudaCheckError() failed : %s\n", cudaGetErrorString(err));
        exit(-1);
    }
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
    const int bottom_length = features.size(2);
    const int top_length = output.size(2);

    return DDFAddFasterForwardLauncher(features, channel_filter, spatial_filter,
                                       kernel_size, dilation, stride,
                                       (kernel_size - 1) * dilation / 2,
                                       batch_size, channels,
                                       bottom_length, top_length,
                                       output);
}
