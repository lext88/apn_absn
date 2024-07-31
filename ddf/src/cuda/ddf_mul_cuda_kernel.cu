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

#define THREADS_PER_BLOCK 1024  // Adjusted for text data
#define MAX_SHARED_MEMORY 49152
#define MAX_SHARED_SCALAR_T 6144
#define kTileDim 32
#define FORWARD_WARP_SIZE 32
#define FORWARD_THREADS_PER_PIXEL 32
#define FULL_MASK 0xffffffff

inline int divideUP(const int x, const int y) { return (((x) + (y)-1) / (y)); }

// Adapt indexing for text data
__device__ inline int Loc2Index(const int n, const int s, const int f,
                                const int seq_len, const int feature_dim) {
    int index = s + n * seq_len * feature_dim + f * seq_len;
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
__device__ __forceinline__ scalar_t warpReduceSum(scalar_t val) {
    for (int offset = 16; offset > 0; offset /= 2)
        val += __shfl_down_sync(FULL_MASK, val, offset);
    return val;
}

// Update to handle text data (sequence length instead of height/width)
template <typename scalar_t>
__global__ void BatchTranspose2DCUDAKernel(const int N, const int S, const int F,
                                           const scalar_t *__restrict__ X,
                                           scalar_t *__restrict__ Y) {
    __shared__ scalar_t tile[kTileDim][kTileDim + 1];
    const int n = blockIdx.x;
    const int s = blockIdx.y;
    const int f = threadIdx.x;

    int x = s * kTileDim + threadIdx.x;
    int y = n * kTileDim + threadIdx.y;
    if (x < S) {
        tile[threadIdx.y][threadIdx.x] = X[Loc2Index(n, s, f, S, F)];
    }
    __syncthreads();

    x = s * kTileDim + threadIdx.x;
    y = n * kTileDim + threadIdx.y;
    if (x < N) {
        Y[Loc2Index(n, s, f, S, F)] = tile[threadIdx.x][threadIdx.y];
    }
}

template <typename scalar_t>
__global__ void DDFForward(const int num_kernels, const scalar_t *__restrict__ bottom_data,
                           const scalar_t *__restrict__ bottom_channel_filter,
                           const scalar_t *__restrict__ bottom_spatial_filter,
                           const int kernel_size, const int dilation,
                           const int stride, const int features,
                           const int seq_len, scalar_t *__restrict__ top_data) {
    __shared__ scalar_t shared_spatial_filter[MAX_SHARED_SCALAR_T];

    bool valid_index = false;
    int index = threadIdx.x + blockIdx.x * blockDim.x;
    if (index >= num_kernels) {
        return;
    }

    const int pixel_id = threadIdx.x / FORWARD_THREADS_PER_PIXEL;
    const int split_id = threadIdx.x % FORWARD_THREADS_PER_PIXEL;
    index = index / FORWARD_THREADS_PER_PIXEL;
    const int seq_pos = index % seq_len;
    const int n = blockIdx.y;

    const int start_pos = seq_pos * stride - ((kernel_size - 1) / 2)*dilation;
    const int end_pos = seq_pos * stride + ((kernel_size - 1) / 2)*dilation + 1;

    scalar_t output_val = 0;
    scalar_t lost = 0;
    scalar_t t = 0;
    scalar_t input = 0;

    int pos, filter_id, feature_id;

    for (int k = split_id; k < kernel_size * kernel_size; k += FORWARD_THREADS_PER_PIXEL) {
        filter_id = Loc2Index(n, k, seq_pos, seq_len, kernel_size * kernel_size);
        shared_spatial_filter[k * FORWARD_WARP_SIZE + pixel_id] = bottom_spatial_filter[filter_id];
    }
    __syncthreads();

    #pragma unroll
    for (int f = split_id; f < features; f += FORWARD_THREADS_PER_PIXEL) {
        output_val = 0;
        lost = 0;
        t = 0;
        input = 0;
        #pragma unroll
        for (pos = start_pos; pos < end_pos; pos += dilation) {
            if (pos < 0 || pos >= seq_len) {
                continue;
            }
            filter_id = (pos - start_pos) / dilation;
            feature_id = Loc2Index(n, f, pos, seq_len, features);
            input = bottom_data[feature_id] * shared_spatial_filter[filter_id * FORWARD_WARP_SIZE + pixel_id];
            t = output_val + input;
            lost += fabs(output_val) >= fabs(input) ? (output_val - t) + input
                                                    : (input - t) + output_val;
            output_val = t;
        }

        const int top_id = Loc2Index(n, f, seq_pos, seq_len, features);
        top_data[top_id] = output_val + lost;
    }
}

int DDFMulForwardLauncher(const at::Tensor features, const at::Tensor channel_filter,
                          const at::Tensor spatial_filter, const int kernel_size,
                          const int dilation, const int stride,
                          const int batch_size, const int features,
                          const int seq_len, at::Tensor output) {
    cudaStream_t stream = at::cuda::getCurrentCUDAStream();
    AT_DISPATCH_FLOATING_TYPES_AND_HALF(
        features.type(), "DDFForward", ([&] {
            const int num_kernels = seq_len * features;
            dim3 grid(batch_size, at::cuda::ATenCeilDiv(num_kernels, THREADS_PER_BLOCK));
            const scalar_t *bottom_data = features.data<scalar_t>();
            const scalar_t *bottom_channel_filter = channel_filter.data<scalar_t>();
            const scalar_t *bottom_spatial_filter = spatial_filter.data<scalar_t>();
            scalar_t *top_data = output.data<scalar_t>();
            DDFForward<scalar_t>
                <<<grid, THREADS_PER_BLOCK, 0, stream>>>(
                num_kernels, bottom_data, bottom_channel_filter,
                bottom_spatial_filter, kernel_size, dilation, stride,
                features, seq_len, top_data);
    }));
    cudaError_t err = cudaGetLastError();
    if (cudaSuccess != err) {
        fprintf(stderr, "cudaCheckError() failed : %s\n", cudaGetErrorString(err));
        exit(-1);
    }
    return 1;
}

// Similarly adapt the backward pass functions as needed.
