import torch
import gather
import cwarp_rnnt._C as core
from typing import Optional, Any, Literal
from pkg_resources import get_distribution

__version__ = get_distribution('cwarp_rnnt').version


class _TransLoss(torch.autograd.Function):
    @staticmethod
    def forward(ctx: Any, xs, lx, ys, ly, blank):
        if xs.requires_grad:
            costs, grads, alphas, betas = core._cwarp_rnnt_forward(
                xs, lx, ys, ly, blank)
            ctx.save_for_backward(grads, ys, lx, ly)
            ctx.blank = blank
            ctx.v_size = xs.size(-1)
        else:
            costs = core._cwarp_rnnt_loss(xs, lx, ys, ly, blank)

        return costs

    @staticmethod
    def backward(ctx: Any, grad_outputs):
        grads, ys, lx, ly = ctx.saved_tensors
        grad_input = core._cwarp_rnnt_backward(
            grads, grad_outputs, ys, lx, ly, ctx.blank, ctx.v_size)
        return grad_input, None, None, None, None


def transducer_loss(
        x: torch.Tensor,
        lx: torch.IntTensor,
        y: torch.IntTensor,
        ly: torch.IntTensor,
        blank: int = 0,
        reduction: Optional[Literal['mean', 'sum', 'none']] = 'mean') -> torch.Tensor:
    """ Calculate Transducer loss with compact layout.
    """
    assert lx.dtype == torch.int32
    assert ly.dtype == torch.int32
    assert y.dtype == torch.int32
    assert reduction in ['mean', 'sum', 'none'] or reduction is None
    # y: (N, U)
    assert y.dim() == 2
    assert y.size(0) == lx.size(0)
    assert y.size(0) == ly.size(0)
    assert y.size(1) == ly.max()
    assert blank >= 0 and blank < x.size(-1)

    if x.dim() == 4:
        print("WARNING: it seems your input is not compact"
              "... would try to compact the input.")
        # suppose x is contiguous in memory layout
        N, T, Up, V = x.size()
        assert y.size(1) + 1 == Up
        assert lx.max() == T
        x = gather.cat(x.view(N, -1, V), lx*(ly+1))

    # x: (STU, V)
    assert x.dim() == 2

    costs = _TransLoss.apply(x, lx, y, ly, blank)

    if reduction == 'mean':
        return costs.mean()
    elif reduction == 'sum':
        return costs.sum()
    elif reduction == 'none' or reduction is None:
        return costs
    else:
        raise ValueError
