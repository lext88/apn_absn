#include <ATen/ATen.h>
#include <ATen/TensorUtils.h>
#include <ATen/Utils.h>
#include <ATen/cuda/CUDAContext.h>
#include <ATen/cuda/CUDAApplyUtils.cuh>
#include <THC/THCAtomics.cuh>
#include <cmath>

using namespace at;  // temporal fix for pytorch<=0.4.1 (see #9848)

#define CUDA_1D_KERNEL_LOOP(i, n)                            \
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n; \
        i += blockDim.x * gridDim.x)

#define THREADS_PER_BLOCK 1024  // Number of threads per block
#define THREADS_PER_EMBEDDING 32
#define MAX_SHARED_MEMORY 49152
#define DATA_TILE 16
#define MAX_KS 4
#define CHANNEL_THREADS 4
#define CHANNEL_BLOCKS 8
#define FULL_MASK 0xffffffff

inline int divideUP(const int x, const int y) { return (((x) + (y)-1) / (y)); }

__device__ inline int Loc2Index(const int n, const int c, const int pos,
                                const int embedding_dim, const int sequence_length,
                                const int channels) {
    return pos + (c + n * channels) * sequence_length;
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
                           const int sequence_length,
                           scalar_t *__restrict__ top_data) {
    __shared__ scalar_t shared_spatial_filter[DATA_TILE * MAX_KS];
    __shared__ scalar_t shared_channel_filter[CHANNEL_THREADS * MAX_KS];
    __shared__ scalar_t shared_data[CHANNEL_THREADS * DATA_TILE];

    const int b = blockIdx.z / CHANNEL_BLOCKS;
    const int cb_id = blockIdx.z % CHANNEL_BLOCKS;
    bool valid_index = false;

    int pos = -999999;

    if ((threadIdx.x - padding) % stride == 0) {
        pos = (threadIdx.x - padding) / stride;
    }
    if (pos >= 0 && pos < sequence_length) {
        valid_index = true;
    }

    const int start_pos = pos * stride;
    const int end_pos = start_pos + kernel_size * dilation;

    const int bottom_pos = blockIdx.x * DATA_TILE * stride - padding + threadIdx.x;

    if (valid_index) {
        if (pos < sequence_length) {
            for (int i = threadIdx.z; i < kernel_size; i += CHANNEL_THREADS) {
                int spatial_filter_id = Loc2Index(b, i, pos, kernel_size, sequence_length, channels);
                shared_spatial_filter[threadIdx.x * kernel_size + i] =
                    bottom_spatial_filter[spatial_filter_id];
            }
        } else {
            for (int i = threadIdx.z; i < kernel_size; i += CHANNEL_THREADS) {
                shared_spatial_filter[threadIdx.x * kernel_size + i] = 0;
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

        if (bottom_pos >= 0 && bottom_pos < sequence_length) {
            int id = Loc2Index(b, c, bottom_pos, sequence_length, channels, sequence_length);
            shared_data[threadIdx.z * DATA_TILE + threadIdx.x] = bottom_data[id];
        } else {
            shared_data[threadIdx.z * DATA_TILE + threadIdx.x] = 0;
        }
        __syncthreads();

        if (valid_index && pos < sequence_length) {
            scalar_t output_val = 0;
            scalar_t lost = 0;
            scalar_t t = 0;
            scalar_t input = 0;

            #pragma unroll
            for (int p = start_pos; p < end_pos; p += dilation) {
                int kernel_p = (p - start_pos) / dilation;

                input = shared_data[threadIdx.z * DATA_TILE + p] *
                    shared_spatial_filter[threadIdx.x * kernel_size + kernel_p] *
                    shared_channel_filter[threadIdx.z * kernel_size + kernel_p];

                t = output_val + input;
                lost += fabs(output_val) >= fabs(input) ? (output_val - t) + input
                                                        : (input - t) + output_val;
                output_val = t;
            }

            int top_id = Loc2Index(b, c, pos, sequence_length, channels);
            top_data[top_id] = output_val + lost;
        }
    }
}

int DDFMulFasterForwardLauncher(const at::Tensor features, const at::Tensor channel_filter,
                                const at::Tensor spatial_filter, const int kernel_size,
                                const int dilation, const int stride,
                                const int batch_size, const int channels,
                                const int sequence_length,
                                at::Tensor output) {
    cudaStream_t stream = at::cuda::getCurrentCUDAStream();
    const int padding = (kernel_size - 1) * dilation / 2;
    const int top_TileDim = divideUP(DATA_TILE - padding * 2, stride);
    const int blocks_x = divideUP(sequence_length, top_TileDim);
    const int blocks_z = batch_size * CHANNEL_BLOCKS;
    dim3 grid(blocks_x, 1, blocks_z);
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
                channels, sequence_length, top_data);
    }));
    cudaError_t err = cudaGetLastError();
    if (cudaSuccess != err) {
        fprintf(stderr, "cudaCheckError() failed : %s\n", cudaGetErrorString(err));
        exit(-1);
    }
    return 1;
}
