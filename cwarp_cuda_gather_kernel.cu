#include <torch/extension.h>

#include <cuda.h>
#include <cuda_runtime.h>

#include <vector>

/* 
    Select the 2 elements from given compact layout input.

    inputs: [STU, V] input in compact layout (w/o paddings)
    index:  [N, U]   a.k.a. target labels
    lx:     [N, ]    Ti of each seq in inputs
    ly:     [N, ]    Ui of each seq in inputs (also for index)
    offset: [N, ]    offset for start location of each seq, this can be calculated from lx and ly
    blank:  int      index of blank label
    output: [STU, 2] output in compact layout
 */
template <typename scalar_t>
__global__ void gather_cuda_kernel(
    const torch::PackedTensorAccessor32<scalar_t, 2, torch::RestrictPtrTraits> inputs,
    const torch::PackedTensorAccessor32<int, 2, torch::RestrictPtrTraits> index,
    const torch::PackedTensorAccessor32<int, 1, torch::RestrictPtrTraits> lx,
    const torch::PackedTensorAccessor32<int, 1, torch::RestrictPtrTraits> ly,
    const torch::PackedTensorAccessor32<int, 1, torch::RestrictPtrTraits> offset,
    const int blank,
    torch::PackedTensorAccessor32<scalar_t, 2, torch::RestrictPtrTraits> output)
{
    unsigned int n = blockIdx.z;
    unsigned int ix = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int iy = blockIdx.y * blockDim.y + threadIdx.y;

    if (ix >= lx[n] || iy > ly[n])
        return;

    unsigned int _index = offset[n] + ix * (ly[n] + 1) + iy;
    // fill blank value
    output[_index][0] = inputs[_index][blank];
    if (iy == ly[n])
        return;
    // fill label value from given index
    output[_index][1] = inputs[_index][index[n][iy]];
}

/* 
    Backward function of gather_cuda_kernel()

    grad_out: [STU, 2] gradients w.r.t. gather output
    grad:     [STU, 2] gradients w.r.t. input of gather_cuda_kernel()
    *: refer to gather_cuda_kernel() for other arguments
 */
template <typename scalar_t>
__global__ void scatter_grad_cuda_kernel(
    const torch::PackedTensorAccessor32<scalar_t, 2, torch::RestrictPtrTraits> grad_out,
    const torch::PackedTensorAccessor32<int, 2, torch::RestrictPtrTraits> index,
    const torch::PackedTensorAccessor32<int, 1, torch::RestrictPtrTraits> lx,
    const torch::PackedTensorAccessor32<int, 1, torch::RestrictPtrTraits> ly,
    const torch::PackedTensorAccessor32<int, 1, torch::RestrictPtrTraits> offset,
    const int blank,
    torch::PackedTensorAccessor32<scalar_t, 2, torch::RestrictPtrTraits> grad)
{
    unsigned int n = blockIdx.z;
    unsigned int ix = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int iy = blockIdx.y * blockDim.y + threadIdx.y;

    if (ix >= lx[n] || iy > ly[n])
        return;

    unsigned int _index = offset[n] + ix * (ly[n] + 1) + iy;
    grad[_index][blank] = grad_out[_index][0];
    if (iy == ly[n])
        return;
    grad[_index][index[n][iy]] = grad_out[_index][1];
}

torch::Tensor gather_cuda_forward(
    torch::Tensor xs,
    torch::Tensor lx,
    torch::Tensor labels,
    torch::Tensor ly,
    torch::Tensor offset,
    int blank)
{
    const auto N = lx.size(0);
    const auto T = lx.max().item<int64_t>();
    const auto Up = ly.max().item<int64_t>() + 1;
    auto output = torch::zeros({xs.size(0), 2}, xs.options());

    dim3 thread(128, 8);
    dim3 block((T + 128 - 1) / 128, (Up + 8 - 1) / 8, N);

    AT_DISPATCH_FLOATING_TYPES(
        xs.scalar_type(),
        "gather_forward_cuda",
        ([&]
         { gather_cuda_kernel<scalar_t><<<block, thread>>>(
               xs.packed_accessor32<scalar_t, 2, torch::RestrictPtrTraits>(),
               labels.packed_accessor32<int, 2, torch::RestrictPtrTraits>(),
               lx.packed_accessor32<int, 1, torch::RestrictPtrTraits>(),
               ly.packed_accessor32<int, 1, torch::RestrictPtrTraits>(),
               offset.packed_accessor32<int, 1, torch::RestrictPtrTraits>(),
               blank,
               output.packed_accessor32<scalar_t, 2, torch::RestrictPtrTraits>()); }));

    const auto err = cudaGetLastError();
    TORCH_CHECK(err == cudaSuccess, "gather forward error: " + std::string(cudaGetErrorString(err)));
    return output;
}

torch::Tensor gather_cuda_backward_(
    torch::Tensor grad_in,
    torch::Tensor grad_out,
    torch::Tensor lx,
    torch::Tensor index,
    torch::Tensor ly,
    torch::Tensor offset,
    int blank)
{
    const auto N = lx.size(0);
    const auto T = lx.max().item<int64_t>();
    const auto Up = ly.max().item<int64_t>() + 1;

    dim3 thread(128, 8);
    dim3 block((T + 128 - 1) / 128, (Up + 8 - 1) / 8, N);
    AT_DISPATCH_FLOATING_TYPES(
        grad_out.scalar_type(),
        "gather_backward_cuda",
        ([&]
         { scatter_grad_cuda_kernel<scalar_t><<<block, thread>>>(
               grad_out.packed_accessor32<scalar_t, 2, torch::RestrictPtrTraits>(),
               index.packed_accessor32<int, 2, torch::RestrictPtrTraits>(),
               lx.packed_accessor32<int, 1, torch::RestrictPtrTraits>(),
               ly.packed_accessor32<int, 1, torch::RestrictPtrTraits>(),
               offset.packed_accessor32<int, 1, torch::RestrictPtrTraits>(),
               blank,
               grad_in.packed_accessor32<scalar_t, 2, torch::RestrictPtrTraits>()); }));

    const auto err = cudaGetLastError();
    TORCH_CHECK(err == cudaSuccess, "Gather backward error: " + std::string(cudaGetErrorString(err)));

    return grad_in;
}