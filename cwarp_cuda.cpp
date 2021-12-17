
#include <torch/extension.h>

#include <tuple>
#include <vector>

// CUDA interface
torch::Tensor gather_cuda_forward(
    torch::Tensor xs,
    torch::Tensor lx,
    torch::Tensor labels,
    torch::Tensor ly,
    torch::Tensor offset,
    int blank);

torch::Tensor gather_cuda_backward_(
    torch::Tensor grad_in,
    torch::Tensor grad_out,
    torch::Tensor lx,
    torch::Tensor labels,
    torch::Tensor ly,
    torch::Tensor offset,
    int blank);

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor>
cwarp_cuda_costs_with_grad(
    const torch::Tensor log_probs,
    const torch::Tensor labels,
    const torch::Tensor lx,
    const torch::Tensor ly,
    const torch::Tensor offset);

torch::Tensor cwarp_cuda_backward_(
    torch::Tensor grad_in,
    const torch::Tensor grad_out,
    const torch::Tensor lx,
    const torch::Tensor ly);

#define CHECK_CUDA(x) TORCH_CHECK(x.device().is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
#define CHECK_TYPE(x, T) TORCH_CHECK(x.scalar_type() == T, #x " must be " #T)
#define CHECK_INPUT(x, T) \
    CHECK_CUDA(x);        \
    CHECK_CONTIGUOUS(x);  \
    CHECK_TYPE(x, T)

#define None torch::indexing::None
#define Slice torch::indexing::Slice

// C++ interface
std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor>
compact_rnnt_loss_forward(
    const torch::Tensor xs,
    const torch::Tensor lx,
    const torch::Tensor ys,
    const torch::Tensor ly,
    const int blank)
{
    CHECK_CONTIGUOUS(xs);
    CHECK_CUDA(xs);
    CHECK_INPUT(lx, torch::ScalarType::Int);
    CHECK_INPUT(ys, torch::ScalarType::Int);
    CHECK_INPUT(ly, torch::ScalarType::Int);

    auto offset = torch::zeros_like(lx);
    {
        auto cumsum = torch::cumsum((lx * (ly + 1)), /*dim=*/0);
        // offset[1:] = cumsum[:-1]
        offset.index_put_({Slice(1, None, None)}, cumsum.index({Slice(0, -1, None)}));
    }

    auto gathered_xs = gather_cuda_forward(xs, lx, ys, ly, offset, blank);

    return cwarp_cuda_costs_with_grad(gathered_xs, ys, lx, ly, offset);
}

torch::Tensor compact_rnnt_loss_backward_(
    torch::Tensor grad_in,
    const torch::Tensor grad_costs,
    const torch::Tensor labels,
    const torch::Tensor lx,
    const torch::Tensor ly,
    const int blank, const int V)
{
    CHECK_CUDA(grad_costs);

    CHECK_INPUT(labels, torch::ScalarType::Int);
    CHECK_INPUT(lx, torch::ScalarType::Int);
    CHECK_INPUT(ly, torch::ScalarType::Int);

    auto offset = torch::zeros_like(lx);
    {
        auto cumsum = torch::cumsum((lx * (ly + 1)), /*dim=*/0);
        // offset[1:] = cumsum[:-1]
        offset.index_put_({Slice(1, None, None)}, cumsum.index({Slice(0, -1, None)}));
    }

    // grad_rnnt and grad_in share the common memory.
    auto grad_rnnt = cwarp_cuda_backward_(grad_in, grad_costs, lx, ly);

    auto grad_gather = torch::zeros({grad_in.size(0), V}, grad_in.options());
    return gather_cuda_backward_(grad_gather, grad_rnnt, lx, labels, ly, offset, blank);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m)
{
    m.def(
        "_cwarp_rnnt_forward",
        &compact_rnnt_loss_forward,
        "CUDA-Warp Transducer loss forward for compact layout.");
    m.def(
        "_cwarp_rnnt_backward",
        &compact_rnnt_loss_backward_,
        "CUDA-Warp Transducer loss backward for compact layout.");
}
