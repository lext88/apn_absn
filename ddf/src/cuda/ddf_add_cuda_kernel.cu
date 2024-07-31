#include <ATen/ATen.h>
#include <ATen/TensorUtils.h>
#include <ATen/Utils.h>
#include <ATen/cuda/CUDAContext.h>
#include <ATen/cuda/CUDAApplyUtils.cuh>
#include <THC/THCAtomics.cuh>
#include <cmath>

using namespace at;

#define CUDA_1D_KERNEL_LOOP(i, n)                            \
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n; \
        i += blockDim.x * gridDim.x)

#define THREADS_PER_BLOCK 1024
#define WARP_SIZE 32
#define THREADS_PER_SEQUENCE 32
#define MAX_SHARED_MEMORY 49152
#define MAX_SHARED_SCALAR_T 6144
#define kTileDim 32
#define kBlockRows 8
#define FORWARD_WARP_SIZE 32
#define FORWARD_THREADS_PER_SEQUENCE 32
#define FULL_MASK 0xffffffff

inline int divideUP(const int x, const int y) { return (((x) + (y)-1) / (y)); }

/* Adjust Loc2Index for 1D sequence data */
__device__ inline int Loc2Index(const int n, const int c, const int p, 
                                const int channel_num, const int length) {
    int index = p + (c + n * channel_num) * length;
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

/* Adjust kernel for 1D sequences */
template <typename scalar_t>
__global__ void BatchTranspose1DCUDAKernel(const int N, const int L, 
                                           const int dl, 
                                           const scalar_t *__restrict__ X, 
                                           scalar_t *__restrict__ Y) {
    __shared__ scalar_t tile[kTileDim];
    const int n = blockIdx.x / dl;
    const int k = blockIdx.x % dl;
    const int offset = n * L;
    int x = k * kTileDim + threadIdx.x;
    int y = threadIdx.x;

    if (x < L) {
        tile[threadIdx.x] = X[offset + x];
    }
    __syncthreads();
    
    x = k * kTileDim + threadIdx.x;
    if (x < L) {
        Y[offset + x] = tile[threadIdx.x];
    }
}

template <typename scalar_t>
__global__ void DDFForward(const int num_kernels, const scalar_t *__restrict__ bottom_data,
                           const scalar_t *__restrict__ bottom_channel_filter,
                           const scalar_t *__restrict__ bottom_spatial_filter,
                           const int kernel_size, const int dilation,
                           const int stride, const int channels,
                           const int bottom_length, const int top_length,
                           scalar_t *__restrict__ top_data) {
    __shared__ scalar_t shared_spatial_filter[MAX_SHARED_SCALAR_T];

    bool valid_index = false;
    int index = threadIdx.x + blockIdx.y * blockDim.x;
    if (index > num_kernels - 1){
        return;
    }

    const int seq_id = threadIdx.x / FORWARD_THREADS_PER_SEQUENCE; // sequence in block from 0 to 15
    const int split_id = threadIdx.x % FORWARD_THREADS_PER_SEQUENCE; // thread in sequence from 0 to 63

    index = index / FORWARD_THREADS_PER_SEQUENCE;
    const int p = index % top_length;
    const int n = blockIdx.x;

    const int start_p = p * stride - ((kernel_size - 1) / 2) * dilation;
    const int end_p = p * stride + ((kernel_size - 1) / 2) * dilation + 1;

    scalar_t output_val = 0;
    scalar_t lost = 0;
    scalar_t t = 0;
    scalar_t input = 0;

    int c, spatial_filter_id, channel_filter_id, bottom_id, top_id;

    for (c = split_id; c < kernel_size * kernel_size; c += FORWARD_THREADS_PER_SEQUENCE) {
        spatial_filter_id = Loc2Index(n, c, p, kernel_size * kernel_size, top_length);
        shared_spatial_filter[c * FORWARD_WARP_SIZE + seq_id] = bottom_spatial_filter[spatial_filter_id];
    }
    __syncthreads();

    #pragma unroll
    for (c = split_id; c < channels; c += FORWARD_THREADS_PER_SEQUENCE) {
        output_val = 0;
        lost = 0;
        t = 0;
        input = 0;
        #pragma unroll
        for (int i = start_p; i < end_p; i += dilation) {
            if (i < 0 || i >= bottom_length) {
                continue;
            }
            int filter_c = (i - start_p) / dilation;
            bottom_id = Loc2Index(n, c, i, channels, bottom_length);

            spatial_filter_id = Loc2Index(n, filter_c, p, kernel_size * kernel_size, top_length);
            channel_filter_id = (n * channels + c ) * kernel_size * kernel_size + filter_c;

            input = bottom_data[bottom_id] *
                    (shared_spatial_filter[filter_c * FORWARD_WARP_SIZE + seq_id] +
                    bottom_channel_filter[channel_filter_id]);

            t = output_val + input;
            lost += fabs(output_val) >= fabs(input) ? (output_val - t) + input
                                                    : (input - t) + output_val;
            output_val = t;
        }

        top_id = Loc2Index(n, c, p, channels, top_length);
        top_data[top_id] = output_val + lost;
    }
}

int DDFAddForwardLauncher(const at::Tensor features, const at::Tensor channel_filter,
                          const at::Tensor spatial_filter, const int kernel_size,
                          const int dilation, const int stride,
                          const int batch_size, const int channels,
                          const int bottom_length, const int top_length,
                          at::Tensor output) {
    cudaStream_t stream = at::cuda::getCurrentCUDAStream();
    AT_DISPATCH_FLOATING_TYPES_AND_HALF(
        features.type(), "DDFForward", ([&] {
            const int num_kernels = top_length * FORWARD_THREADS_PER_SEQUENCE;
            dim3 grid(batch_size, at::cuda::ATenCeilDiv(num_kernels, THREADS_PER_BLOCK));
            const scalar_t *bottom_data = features.data<scalar_t>();
            const scalar_t *bottom_channel_filter = channel_filter.data<scalar_t>();
            const scalar_t *bottom_spatial_filter = spatial_filter.data<scalar_t>();
            scalar_t *top_data = output.data<scalar_t>();
            DDFForward<scalar_t>
                <<<grid, THREADS_PER_BLOCK, 0, stream>>>(
                num_kernels, bottom_data, bottom_channel_filter,
                bottom_spatial_filter, kernel_size, dilation, stride,
                channels, bottom_length, top_length, top_data);
    }));
    cudaError_t err = cudaGetLastError();
    if (cudaSuccess != err) {
        fprintf(stderr, "cudaCheckError() failed : %s\n", cudaGetErrorString(err));
        exit(-1);
    }
    return 1;
}

template <typename scalar_t>
__global__ void DDFBackward_Feature(const int num_kernels, const scalar_t *__restrict__ top_diff,
                                    const scalar_t *__restrict__ bottom_spatial_filter,
                                    const scalar_t *__restrict__ bottom_channel_filter,
                                    const int kernel_size, const int dilation,
                                    const int stride, const int channels,
                                    const int top_length, const int bottom_length,
                                    scalar_t *__restrict__ bottom_diff) {
    __shared__ scalar_t shared_spatial_filter[MAX_SHARED_SCALAR_T];

    int index = threadIdx.x + blockIdx.x * blockDim.x;
    if (index > num_kernels - 1) {
        return;
    }

    const int seq_id = threadIdx.x / THREADS_PER_SEQUENCE;
    const int split_id = threadIdx.x % THREADS_PER_SEQUENCE;

    index = index / THREADS_PER_SEQUENCE;
    const int p = index % bottom_length;
    const int n = index / bottom_length;

    const int start_p = p - ((kernel_size - 1) / 2) * dilation;
    const int end_p = p + ((kernel_size - 1) / 2) * dilation + 1;

    scalar_t output_val = 0;
    scalar_t lost = 0;
    scalar_t t = 0;
    scalar_t input = 0;

    int c, spatial_filter_id, channel_filter_id, bottom_id, top_id;

    for (c = split_id; c < kernel_size * kernel_size; c += THREADS_PER_SEQUENCE) {
        spatial_filter_id = Loc2Index(n, c, p, kernel_size * kernel_size, bottom_length);
        shared_spatial_filter[c * THREADS_PER_SEQUENCE + seq_id] = bottom_spatial_filter[spatial_filter_id];
    }
    __syncthreads();

    #pragma unroll
    for (c = split_id; c < channels; c += THREADS_PER_SEQUENCE) {
        output_val = 0;
        lost = 0;
        t = 0;
        input = 0;
        #pragma unroll
        for (int i = start_p; i < end_p; i += dilation) {
            if (i < 0 || i >= top_length) {
                continue;
            }
            int filter_c = (i - start_p) / dilation;
            top_id = Loc2Index(n, c, i, channels, top_length);

            spatial_filter_id = Loc2Index(n, filter_c, p, kernel_size * kernel_size, bottom_length);
            channel_filter_id = (n * channels + c) * kernel_size * kernel_size + filter_c;

            input = top_diff[top_id] * (shared_spatial_filter[filter_c * THREADS_PER_SEQUENCE + seq_id] +
                                        bottom_channel_filter[channel_filter_id]);

            t = output_val + input;
            lost += fabs(output_val) >= fabs(input) ? (output_val - t) + input
                                                    : (input - t) + output_val;
            output_val = t;
        }

        bottom_id = Loc2Index(n, c, p, channels, bottom_length);
        bottom_diff[bottom_id] = output_val + lost;
    }
}

int DDFAddBackwardLauncher(const at::Tensor top_diff, const at::Tensor channel_filter,
                           const at::Tensor spatial_filter, const int kernel_size,
                           const int dilation, const int stride,
                           const int batch_size, const int channels,
                           const int top_length, const int bottom_length,
                           at::Tensor bottom_diff) {
    cudaStream_t stream = at::cuda::getCurrentCUDAStream();
    AT_DISPATCH_FLOATING_TYPES_AND_HALF(
        top_diff.type(), "DDFBackward_Feature", ([&] {
            const int num_kernels = top_length * THREADS_PER_SEQUENCE;
            dim3 grid(batch_size, at::cuda::ATenCeilDiv(num_kernels, THREADS_PER_BLOCK));
            const scalar_t *top_diff = top_diff.data<scalar_t>();
            const scalar_t *bottom_channel_filter = channel_filter.data<scalar_t>();
            const scalar_t *bottom_spatial_filter = spatial_filter.data<scalar_t>();
            scalar_t *bottom_diff = bottom_diff.data<scalar_t>();
            DDFBackward_Feature<scalar_t>
                <<<grid, THREADS_PER_BLOCK, 0, stream>>>(
                num_kernels, top_diff, bottom_spatial_filter,
                bottom_channel_filter, kernel_size, dilation, stride,
                channels, top_length, bottom_length, bottom_diff);
    }));
    cudaError_t err = cudaGetLastError();
    if (cudaSuccess != err) {
        fprintf(stderr, "cudaCheckError() failed : %s\n", cudaGetErrorString(err));
        exit(-1);
    }
    return 1;
}
