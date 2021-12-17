#include <torch/extension.h>

#include <cuda.h>
#include <cuda_runtime.h>

#include <vector>

#define W 32
#define logSumExp(a, b) log_sum_exp(static_cast<float>(a), static_cast<float>(b))

// conduct logexpsum in float
__device__ __forceinline__ float log_sum_exp(float a, float b)
{
    float maximum, diff;
    maximum = (a > b) ? a : b;
    diff = (a > b) ? (b - a) : (a - b);
    maximum += log1pf(expf(diff));
    return maximum;
}

template <typename scalar_t>
__device__ void cal_alphas_kernel(
    const torch::PackedTensorAccessor32<scalar_t, 2, torch::RestrictPtrTraits> log_probs,
    const torch::PackedTensorAccessor32<int, 2, torch::RestrictPtrTraits> labels,
    const torch::PackedTensorAccessor32<int, 1, torch::RestrictPtrTraits> lx,
    const torch::PackedTensorAccessor32<int, 1, torch::RestrictPtrTraits> ly,
    const torch::PackedTensorAccessor32<int, 1, torch::RestrictPtrTraits> offset,
    torch::PackedTensorAccessor32<int, 3, torch::RestrictPtrTraits> trace_lock,
    torch::PackedTensorAccessor32<scalar_t, 3, torch::RestrictPtrTraits> alphas)
{
    unsigned int n = blockIdx.z * blockDim.z + threadIdx.z;
    if (n >= alphas.size(0))
        return;

    unsigned int d = threadIdx.x;
    unsigned int g = blockIdx.x;
    unsigned int u = blockIdx.y + 1;
    unsigned int p = g * W;
    unsigned int t = p + d + 1;

    unsigned int actual_t = lx[n];
    unsigned int actual_u = ly[n] + 1;

    if (t > actual_t || u > actual_u)
        return;

    int *lock = trace_lock[0][n].data() + blockIdx.y;

    if (blockIdx.x == 0 && blockIdx.y == 0)
    {
        // initialize the state as log(p) = 0.
        alphas[n][0][0] = static_cast<scalar_t>(0.0F);
    }

    if (blockIdx.x > 0)
    {
        // Wait previous row
        do
        {
        } while (atomicAdd(lock, 0) < g);
    }

    if (blockIdx.y > 0)
    {
        // Wait previous column
        do
        {
        } while (atomicAdd(lock - 1, 0) <= g);
    }

    if (blockIdx.x == 0 && u < actual_u)
    {
        alphas[n][0][u] = alphas[n][0][u - 1] + log_probs[offset[n] + u - 1][1];
    }

    if (blockIdx.y == 0 && t < actual_t)
    {
        scalar_t a;
        scalar_t b = log_probs[offset[n] + (t - 1) * actual_u][0];

#pragma unroll
        for (unsigned int i = 1; i < W; i *= 2)
        {
            a = __shfl_up_sync(0xffffffff, b, i);
            if (i <= d)
            {
                b += a;
            }
        }

        a = alphas[n][p][0];
        alphas[n][t][0] = a + b;
    }

    if (t < actual_t && u < actual_u)
    {
        scalar_t bias = log_probs[offset[n] + (t - 1) * actual_u + u][0];
        scalar_t skip = alphas[n][p][u] + bias;
        scalar_t emit = alphas[n][t][u - 1] + log_probs[offset[n] + t * actual_u + u - 1][1];

        scalar_t r = logSumExp(skip, emit);
        scalar_t output = r;

        for (unsigned int i = 1; i < W; i++)
        {
            r = __shfl_up_sync(0xffffffff, r, 1);
            if (i == d)
            {
                r = logSumExp(r + bias, emit);
                output = r;
            }
        }

        alphas[n][t][u] = output;
    }

    if (d == 0)
    {
        // https://stackoverflow.com/a/5233737
        __threadfence();
        atomicAdd(lock, 1);
    }
}

template <typename scalar_t>
__device__ void cal_betas_kernel(
    const torch::PackedTensorAccessor32<scalar_t, 2, torch::RestrictPtrTraits> log_probs,
    const torch::PackedTensorAccessor32<int, 2, torch::RestrictPtrTraits> labels,
    const torch::PackedTensorAccessor32<int, 1, torch::RestrictPtrTraits> lx,
    const torch::PackedTensorAccessor32<int, 1, torch::RestrictPtrTraits> ly,
    const torch::PackedTensorAccessor32<int, 1, torch::RestrictPtrTraits> offset,
    torch::PackedTensorAccessor32<int, 3, torch::RestrictPtrTraits> trace_lock,
    torch::PackedTensorAccessor32<scalar_t, 3, torch::RestrictPtrTraits> betas)
{
    unsigned int n = blockIdx.z * blockDim.z + threadIdx.z;
    if (n >= betas.size(0))
        return;

    unsigned int d = threadIdx.x;
    unsigned int g = blockIdx.x;
    unsigned int u = blockIdx.y + 1;
    unsigned int p = g * W;
    unsigned int t = p + d + 1;

    unsigned int actual_t = lx[n];
    unsigned int actual_u = ly[n] + 1;

    if (t > actual_t || u > actual_u)
        return;

    int T1 = actual_t - 1;
    int U1 = actual_u - 1;
    int *lock = trace_lock[1][n].data() + blockIdx.y;

    if (blockIdx.x == 0 && blockIdx.y == 0)
    {
        betas[n][T1][U1] = log_probs[offset[n] + T1 * actual_u + U1][0];
    }

    if (blockIdx.x > 0)
    {
        // Wait previous row
        do
        {
        } while (atomicAdd(lock, 0) < g);
    }

    if (blockIdx.y > 0)
    {
        // Wait previous column
        do
        {
        } while (atomicAdd(lock - 1, 0) <= g);
    }

    if (blockIdx.x == 0 && u < actual_u)
    {
        betas[n][T1][U1 - u] = betas[n][T1][U1 - u + 1] +
                               log_probs[offset[n] + T1 * actual_u + U1 - u][1];
    }

    if (blockIdx.y == 0 && t < actual_t)
    {
        scalar_t a;
        scalar_t b = log_probs[offset[n] + (T1 - t) * actual_u + U1][0];

#pragma unroll
        for (unsigned int i = 1; i < W; i *= 2)
        {
            a = __shfl_up_sync(0xffffffff, b, i);
            if (i <= d)
            {
                b += a;
            }
        }

        a = betas[n][T1 - p][U1];
        betas[n][T1 - t][U1] = a + b;
    }
    if (t < actual_t && u < actual_u)
    {
        int tmp_index = offset[n] + (T1 - t) * actual_u + U1 - u;
        scalar_t bias = log_probs[tmp_index][0];
        scalar_t skip = betas[n][T1 - p][U1 - u] + bias;
        scalar_t emit = betas[n][T1 - t][U1 - u + 1] + log_probs[tmp_index][1];

        scalar_t r = logSumExp(skip, emit);
        scalar_t output = r;

        for (unsigned int i = 1; i < W; i++)
        {
            r = __shfl_up_sync(0xffffffff, r, 1);
            if (i == d)
            {
                r = logSumExp(r + bias, emit);
                output = r;
            }
        }

        betas[n][T1 - t][U1 - u] = output;
    }

    if (d == 0)
    {
        // https://stackoverflow.com/a/5233737
        __threadfence();
        atomicAdd(lock, 1);
    }
}

template <typename scalar_t>
__global__ void cal_alpha_beta_kernel(
    const torch::PackedTensorAccessor32<scalar_t, 2, torch::RestrictPtrTraits> log_probs,
    const torch::PackedTensorAccessor32<int, 2, torch::RestrictPtrTraits> labels,
    const torch::PackedTensorAccessor32<int, 1, torch::RestrictPtrTraits> lx,
    const torch::PackedTensorAccessor32<int, 1, torch::RestrictPtrTraits> ly,
    const torch::PackedTensorAccessor32<int, 1, torch::RestrictPtrTraits> offset,
    torch::PackedTensorAccessor32<int, 3, torch::RestrictPtrTraits> trace_lock,
    torch::PackedTensorAccessor32<scalar_t, 3, torch::RestrictPtrTraits> alphas,
    torch::PackedTensorAccessor32<scalar_t, 3, torch::RestrictPtrTraits> betas)
{
    if (threadIdx.y == 0)
    {
        cal_alphas_kernel<scalar_t>(log_probs, labels, lx, ly, offset, trace_lock, alphas);
    }
    else if (threadIdx.y == 1)
    {
        cal_betas_kernel<scalar_t>(log_probs, labels, lx, ly, offset, trace_lock, betas);
    }
}

template <typename scalar_t>
__global__ void fill_grad_blank_kernel(
    const torch::PackedTensorAccessor32<scalar_t, 2, torch::RestrictPtrTraits> log_probs,
    const torch::PackedTensorAccessor32<int, 2, torch::RestrictPtrTraits> labels,
    const torch::PackedTensorAccessor32<int, 1, torch::RestrictPtrTraits> lx,
    const torch::PackedTensorAccessor32<int, 1, torch::RestrictPtrTraits> ly,
    const torch::PackedTensorAccessor32<int, 1, torch::RestrictPtrTraits> offset,
    const torch::PackedTensorAccessor32<scalar_t, 3, torch::RestrictPtrTraits> alphas,
    const torch::PackedTensorAccessor32<scalar_t, 3, torch::RestrictPtrTraits> betas,
    torch::PackedTensorAccessor32<scalar_t, 2, torch::RestrictPtrTraits> grad)
{
    unsigned int u = blockIdx.y;
    unsigned int n = blockIdx.z;
    unsigned int t = blockIdx.x * blockDim.x + threadIdx.x;

    unsigned int actual_t = lx[n];
    unsigned int actual_u = ly[n] + 1;

    if (t >= actual_t || u >= actual_u)
        return;

    if (t == actual_t - 1 && u < actual_u - 1)
        return;

    scalar_t a = alphas[n][t][u];
    if (t < actual_t - 1)
    {
        a += betas[n][t + 1][u];
    }

    // index = (n, t, u);
    int _index = offset[n] + t * actual_u + u;
    grad[_index][0] = -exp(a + log_probs[_index][0] - betas[n][0][0]);
}

template <typename scalar_t>
__global__ void fill_grad_label_kernel(
    const torch::PackedTensorAccessor32<scalar_t, 2, torch::RestrictPtrTraits> log_probs,
    const torch::PackedTensorAccessor32<int, 2, torch::RestrictPtrTraits> labels,
    const torch::PackedTensorAccessor32<int, 1, torch::RestrictPtrTraits> lx,
    const torch::PackedTensorAccessor32<int, 1, torch::RestrictPtrTraits> ly,
    const torch::PackedTensorAccessor32<int, 1, torch::RestrictPtrTraits> offset,
    const torch::PackedTensorAccessor32<scalar_t, 3, torch::RestrictPtrTraits> alphas,
    const torch::PackedTensorAccessor32<scalar_t, 3, torch::RestrictPtrTraits> betas,
    torch::PackedTensorAccessor32<scalar_t, 2, torch::RestrictPtrTraits> grad)
{
    unsigned int u = blockIdx.y;
    unsigned int n = blockIdx.z;
    unsigned int t = blockIdx.x * blockDim.x + threadIdx.x;

    if (t >= lx[n] || u >= ly[n])
        return;

    scalar_t a = alphas[n][t][u] + betas[n][t][u + 1];
    // index = (n, t, u);
    int _index = offset[n] + t * (ly[n] + 1) + u;
    grad[_index][1] = -exp(a + log_probs[_index][1] - betas[n][0][0]);
}

template <typename scalar_t>
__global__ void fill_cost_kernel(
    const torch::PackedTensorAccessor32<scalar_t, 2, torch::RestrictPtrTraits> log_probs,
    const torch::PackedTensorAccessor32<int, 1, torch::RestrictPtrTraits> lx,
    const torch::PackedTensorAccessor32<int, 1, torch::RestrictPtrTraits> ly,
    const torch::PackedTensorAccessor32<int, 1, torch::RestrictPtrTraits> offset,
    const torch::PackedTensorAccessor32<scalar_t, 3, torch::RestrictPtrTraits> alphas,
    const torch::PackedTensorAccessor32<scalar_t, 3, torch::RestrictPtrTraits> betas,
    torch::PackedTensorAccessor32<scalar_t, 1, torch::RestrictPtrTraits> costs)
{

    unsigned int n = blockIdx.x * blockDim.x + threadIdx.x;

    if (n >= alphas.size(0))
        return;

    unsigned int t = lx[n] - 1;
    unsigned int u = ly[n];

    scalar_t a = alphas[n][t][u] + log_probs[offset[n] + t * (ly[n] + 1) + u][0];
    scalar_t b = betas[n][0][0];

    scalar_t ratio = abs(a - b) / abs(max(a, b));

    if (ratio > 0.001)
    {
        // FIXME: maybe we should empty the gradients ?
        printf("\nWARNING: sample %d (%d, %d) has a forward/backward mismatch %f / %f\n",
               n, t + 1, u, a, b);
        b = (a + b) / scalar_t(2.0F);
    }

    costs[n] = -b;
}

/* 
    compute rnnt loss as well as the gradients via forward/backward algorithm.

    args:
        log_probs:  [STU, 2]    log probabilities in compact layout and being gathered
        labels:     [N, U]      labels
        lx:         [N, ]
        ly:         [N, ]
        offset:     [N, ]       start offset of each batch in log_probs, offset[i]-offset[i-1] = lx[i-1] * (ly[i-1]+1) 

    return:
        tuple(costs, grads, alphas, betas)
        costs:      [N, ]       -log P(Y|X)
        grads:      [STU, 2]    gradients to log_probs
        alphas:     [N, T, U+1] fw variables, might be useful for debugging or analysis, see Graves paper.
        betas:      [N, T, U+1] bw variables.
 */
std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor>
cwarp_cuda_costs_with_grad(
    const torch::Tensor log_probs,
    const torch::Tensor labels,
    const torch::Tensor lx,
    const torch::Tensor ly,
    const torch::Tensor offset)
{
    const auto N = lx.size(0);
    const auto T = lx.max().item<int64_t>();      // max of {T_i}
    const auto Up = ly.max().item<int64_t>() + 1; // max of {U_i} + 1

    auto alphas = torch::zeros({N, T, Up}, log_probs.options());
    auto betas = torch::zeros_like(alphas);
    auto grads = torch::zeros_like(log_probs);
    auto costs = torch::empty({N}, log_probs.options());

    auto trace_locks = torch::zeros({2, N, Up - 1}, labels.options());
    unsigned int Wl = 0;
    if (N <= 4)
    {
        Wl = 2;
    }
    else if (N <= 8)
    {
        Wl = 4;
    }
    else if (N <= 16)
    {
        Wl = 8;
    }
    else
    {
        Wl = 16;
    }

    dim3 thread0(W, 2, Wl);
    dim3 block0((T + W - 1) / W, Up, (N + Wl - 1) / Wl);

    AT_DISPATCH_FLOATING_TYPES(
        log_probs.scalar_type(),
        "rnnt_cwarp_cuda",
        ([&]
         { cal_alpha_beta_kernel<scalar_t><<<block0, thread0>>>(
               log_probs.packed_accessor32<scalar_t, 2, torch::RestrictPtrTraits>(),
               labels.packed_accessor32<int, 2, torch::RestrictPtrTraits>(),
               lx.packed_accessor32<int, 1, torch::RestrictPtrTraits>(),
               ly.packed_accessor32<int, 1, torch::RestrictPtrTraits>(),
               offset.packed_accessor32<int, 1, torch::RestrictPtrTraits>(),
               trace_locks.packed_accessor32<int, 3, torch::RestrictPtrTraits>(),
               alphas.packed_accessor32<scalar_t, 3, torch::RestrictPtrTraits>(),
               betas.packed_accessor32<scalar_t, 3, torch::RestrictPtrTraits>()); }));

    auto err = cudaGetLastError();
    TORCH_CHECK(err == cudaSuccess, "alpha/beta calculation error: " + std::string(cudaGetErrorString(err)));

    dim3 block1((T + 1024 - 1) / 1024, Up, N);
    AT_DISPATCH_FLOATING_TYPES(
        log_probs.scalar_type(),
        "rnnt_fill_grad_blank_cuda",
        ([&]
         { fill_grad_blank_kernel<scalar_t><<<block1, 1024>>>(
               log_probs.packed_accessor32<scalar_t, 2, torch::RestrictPtrTraits>(),
               labels.packed_accessor32<int, 2, torch::RestrictPtrTraits>(),
               lx.packed_accessor32<int, 1, torch::RestrictPtrTraits>(),
               ly.packed_accessor32<int, 1, torch::RestrictPtrTraits>(),
               offset.packed_accessor32<int, 1, torch::RestrictPtrTraits>(),
               alphas.packed_accessor32<scalar_t, 3, torch::RestrictPtrTraits>(),
               betas.packed_accessor32<scalar_t, 3, torch::RestrictPtrTraits>(),
               grads.packed_accessor32<scalar_t, 2, torch::RestrictPtrTraits>()); }));

    err = cudaGetLastError();
    TORCH_CHECK(err == cudaSuccess, "fill blank grad error: " + std::string(cudaGetErrorString(err)));

    dim3 block2((T + 1024 - 1) / 1024, Up - 1, N);
    AT_DISPATCH_FLOATING_TYPES(
        log_probs.scalar_type(),
        "rnnt_fill_grad_label_cuda",
        ([&]
         { fill_grad_label_kernel<scalar_t><<<block2, 1024>>>(
               log_probs.packed_accessor32<scalar_t, 2, torch::RestrictPtrTraits>(),
               labels.packed_accessor32<int, 2, torch::RestrictPtrTraits>(),
               lx.packed_accessor32<int, 1, torch::RestrictPtrTraits>(),
               ly.packed_accessor32<int, 1, torch::RestrictPtrTraits>(),
               offset.packed_accessor32<int, 1, torch::RestrictPtrTraits>(),
               alphas.packed_accessor32<scalar_t, 3, torch::RestrictPtrTraits>(),
               betas.packed_accessor32<scalar_t, 3, torch::RestrictPtrTraits>(),
               grads.packed_accessor32<scalar_t, 2, torch::RestrictPtrTraits>()); }));

    err = cudaGetLastError();
    TORCH_CHECK(err == cudaSuccess, "fill label grad error: " + std::string(cudaGetErrorString(err)));

    dim3 block3((N + Wl - 1) / Wl);
    AT_DISPATCH_FLOATING_TYPES(
        log_probs.scalar_type(),
        "rnnt_fill_cost_cuda",
        ([&]
         { fill_cost_kernel<scalar_t><<<block3, Wl>>>(
               log_probs.packed_accessor32<scalar_t, 2, torch::RestrictPtrTraits>(),
               lx.packed_accessor32<int, 1, torch::RestrictPtrTraits>(),
               ly.packed_accessor32<int, 1, torch::RestrictPtrTraits>(),
               offset.packed_accessor32<int, 1, torch::RestrictPtrTraits>(),
               alphas.packed_accessor32<scalar_t, 3, torch::RestrictPtrTraits>(),
               betas.packed_accessor32<scalar_t, 3, torch::RestrictPtrTraits>(),
               costs.packed_accessor32<scalar_t, 1, torch::RestrictPtrTraits>()); }));

    err = cudaGetLastError();
    TORCH_CHECK(err == cudaSuccess, "fill costs error: " + std::string(cudaGetErrorString(err)));

    return std::make_tuple(costs, grads, alphas, betas);
}

/*
    backward function of cwarp_cuda_costs_with_grad()
    ### This is a in-place function ! ###

    args:
        grad_in:    [STU, 2]    the return value `grad` of cwarp_cuda_costs_with_grad()
        grad_out:   [N, ]       gradients of costs.
        lx:         [N, ]
        ly:         [N, ]
    
    return:
        grad_in
 */
torch::Tensor cwarp_cuda_backward_(
    torch::Tensor grad_in,
    const torch::Tensor grad_out,
    const torch::Tensor lx,
    const torch::Tensor ly)
{
    const auto expand_grad_out = torch::repeat_interleave(grad_out, lx * (ly + 1));
    grad_in *= expand_grad_out.view({-1, 1});
    return grad_in;
}