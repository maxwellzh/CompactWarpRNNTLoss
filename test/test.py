import torch
import sys
import cwarp_rnnt._C as core
import warp_rnnt as base
import cwarp_rnnt as crnnt

from typing import Union, Tuple


def generate_data(Nmax: int = 20, Tmax: int = 512, Umax: int = 512, Vmax: int = 128) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:

    Umax += 1
    N = torch.randint(1, Nmax, (1,)).item()
    T = torch.randint(1, Tmax, (1,)).item()
    U = torch.randint(2, Umax, (1,)).item()
    V = torch.randint(2, Vmax, (1,)).item()
    V = 3

    xs = torch.randn((N, T, U, V), dtype=torch.float32,
                     device=0).log_softmax(dim=-1)
    ys = torch.randint(1, V, (N, U-1), dtype=torch.int, device=0)
    xn = torch.randint(max(1, T // 2), T+1, (N,), dtype=torch.int, device=0)
    yn = torch.randint(max(1, U // 2), U, (N,), dtype=torch.int, device=0)
    xn = xn + T - xn.max()
    yn = yn + U-1 - yn.max()

    ys = ys.to(dtype=torch.int)
    xn, yn = xn.to(dtype=torch.int, device=0), yn.to(
        dtype=torch.int, device=0)

    return xs, xn, ys, yn


def compactTensor(xs: torch.Tensor, ys: torch.Tensor, xn: torch.Tensor, yn: torch.Tensor):

    assert xs.dim() == 4
    assert ys.dim() == 2

    N, T, Up, V = xs.size()
    assert ys.size() == (N, Up-1)
    assert xn.size(0) == N
    assert yn.size(0) == N

    _ys = torch.cat([ys[i, :yn[i]] for i in range(N)])
    _xs = [xs[i, :xn[i], :yn[i]+1, :].contiguous() for i in range(N)]
    _xs = torch.cat([x.view(-1, V) for x in _xs], dim=0)

    return _xs, _ys


if __name__ == "__main__":
    if sys.argv[1:] != []:
        SEED = int(sys.argv[1])
    else:
        SEED = 0
    torch.manual_seed(SEED)
    torch.cuda.manual_seed(SEED)
    torch.backends.cudnn.deterministic = True

    xs, lx, ys, ly = generate_data()  # 4, 6, 6, 4)
    print(xs.size())
    print(ys.size())
    print("x lens range:", lx.min().item(), lx.max().item())
    print("y lens range:", ly.min().item(), ly.max().item())
    xs, _ys = compactTensor(xs, ys, lx, ly)

    ref_cost = base.rnnt_loss(xs, _ys, lx, ly, gather=True, compact=True)
    cal_cost = crnnt.transducer_loss(xs, lx, ys, ly, 0, 'none')
    print((ref_cost - cal_cost) < 1e-3)

    exit(0)

    # print("x")
    # print(xs)
    # print("y")
    # print(ys)
    # print(lx)
    # print(ly)

    cwarp_cost, grads, alphas, betas = core._cwarp_rnnt_forward(
        xs, lx, ys, ly, 0)
    cost_wo_grad = core._cwarp_rnnt_loss(xs, lx, ys, ly, 0)
    print(torch.all((cost_wo_grad - cwarp_cost) < 1e-3))
    print("Forward match: ", torch.all(
        (ref_cost-cwarp_cost).abs() / max(ref_cost.abs().max(), cwarp_cost.abs().max()) < 1e-4).item())

    exit(0)
    xs.requires_grad = True
    ref_cost = base.rnnt_loss(xs, _ys, lx, ly, gather=True, compact=True)
    ref_cost.sum().backward()
    ref_grad = xs.grad.data
    xs.grad = None

    cwarp_cost, grads, alphas, betas = core._cwarp_rnnt_forward(
        xs, lx, ys, ly, 0)

    grad_cwarp = core._cwarp_rnnt_backward(grads, cwarp_cost.new_ones(
        cwarp_cost.size(0)), ys, lx, ly, 0, xs.size(-1))

    print("Backward match: ", torch.all(
        (ref_grad-grad_cwarp).abs() < 1e-3).item())
