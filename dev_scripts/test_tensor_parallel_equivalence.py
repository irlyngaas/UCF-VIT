#!/usr/bin/env python3
"""Exact CPU equivalence checks for the custom tensor-parallel layers."""

import os
import tempfile

import torch
import torch.distributed as dist
import torch.multiprocessing as mp

from UCF_VIT.fsdp.building_blocks import Attention, Mlp
from UCF_VIT.utils.fused_attn import FusedAttn


TP_SIZE = 8
ATOL = 2e-5
RTOL = 2e-5


def _assert_close(name, actual, expected):
    try:
        torch.testing.assert_close(actual, expected, atol=ATOL, rtol=RTOL)
    except AssertionError as exc:
        raise AssertionError(f"{name} differs between TP1 and TP{TP_SIZE}") from exc


def _copy_mlp_shard(full, shard, rank):
    with torch.no_grad():
        shard.fc1.weight.copy_(full.fc1.weight.chunk(TP_SIZE, dim=0)[rank])
        shard.fc1.bias.copy_(full.fc1.bias.chunk(TP_SIZE, dim=0)[rank])
        shard.fc2.weight.copy_(full.fc2.weight.chunk(TP_SIZE, dim=1)[rank])
        shard.fc2.bias.copy_(full.fc2.bias)


def _copy_attention_shard(full, shard, rank):
    with torch.no_grad():
        qkv_weight = full.qkv.weight.reshape(3, -1, full.qkv.in_features)
        local_qkv_weight = torch.cat(
            [part.chunk(TP_SIZE, dim=0)[rank] for part in qkv_weight], dim=0
        )
        shard.qkv.weight.copy_(local_qkv_weight)

        qkv_bias = full.qkv.bias.reshape(3, -1)
        local_qkv_bias = torch.cat(
            [part.chunk(TP_SIZE, dim=0)[rank] for part in qkv_bias], dim=0
        )
        shard.qkv.bias.copy_(local_qkv_bias)

        shard.proj.weight.copy_(full.proj.weight.chunk(TP_SIZE, dim=1)[rank])
        shard.proj.bias.copy_(full.proj.bias)


def _check_mlp(rank):
    torch.manual_seed(100)
    full = Mlp(in_features=32, hidden_features=64, out_features=32, drop=0.0)
    shard = Mlp(
        in_features=32,
        hidden_features=64,
        out_features=32,
        drop=0.0,
        tensor_par_size=TP_SIZE,
        tensor_par_group=dist.group.WORLD,
    )
    _copy_mlp_shard(full, shard, rank)

    torch.manual_seed(101)
    full_input = torch.randn(2, 5, 32, requires_grad=True)
    shard_input = full_input.detach().clone().requires_grad_(True)
    upstream = torch.randn(2, 5, 32)

    full_output = full(full_input)
    shard_output = shard(shard_input)
    _assert_close("MLP output", shard_output, full_output)

    full_output.backward(upstream)
    shard_output.backward(upstream)
    _assert_close("MLP input gradient", shard_input.grad, full_input.grad)
    _assert_close("MLP fc1 weight gradient", shard.fc1.weight.grad,
                  full.fc1.weight.grad.chunk(TP_SIZE, dim=0)[rank])
    _assert_close("MLP fc1 bias gradient", shard.fc1.bias.grad,
                  full.fc1.bias.grad.chunk(TP_SIZE, dim=0)[rank])
    _assert_close("MLP fc2 weight gradient", shard.fc2.weight.grad,
                  full.fc2.weight.grad.chunk(TP_SIZE, dim=1)[rank])
    _assert_close("MLP output bias gradient", shard.fc2.bias.grad, full.fc2.bias.grad)


def _check_attention(rank):
    torch.manual_seed(200)
    full = Attention(
        dim=32,
        num_heads=8,
        qkv_bias=True,
        fused_attn=FusedAttn.DEFAULT,
    )
    shard = Attention(
        dim=32,
        num_heads=8,
        qkv_bias=True,
        fused_attn=FusedAttn.DEFAULT,
        tensor_par_size=TP_SIZE,
        tensor_par_group=dist.group.WORLD,
    )
    _copy_attention_shard(full, shard, rank)

    torch.manual_seed(201)
    full_input = torch.randn(2, 5, 32, requires_grad=True)
    shard_input = full_input.detach().clone().requires_grad_(True)
    upstream = torch.randn(2, 5, 32)

    full_output = full(full_input)
    shard_output = shard(shard_input)
    _assert_close("attention output", shard_output, full_output)

    full_output.backward(upstream)
    shard_output.backward(upstream)
    _assert_close("attention input gradient", shard_input.grad, full_input.grad)

    full_qkv_weight_grad = full.qkv.weight.grad.reshape(3, -1, 32)
    expected_qkv_weight_grad = torch.cat(
        [part.chunk(TP_SIZE, dim=0)[rank] for part in full_qkv_weight_grad], dim=0
    )
    _assert_close("attention qkv weight gradient", shard.qkv.weight.grad,
                  expected_qkv_weight_grad)

    full_qkv_bias_grad = full.qkv.bias.grad.reshape(3, -1)
    expected_qkv_bias_grad = torch.cat(
        [part.chunk(TP_SIZE, dim=0)[rank] for part in full_qkv_bias_grad], dim=0
    )
    _assert_close("attention qkv bias gradient", shard.qkv.bias.grad,
                  expected_qkv_bias_grad)
    _assert_close("attention projection weight gradient", shard.proj.weight.grad,
                  full.proj.weight.grad.chunk(TP_SIZE, dim=1)[rank])
    _assert_close("attention output bias gradient", shard.proj.bias.grad,
                  full.proj.bias.grad)


def _worker(rank, rendezvous_path):
    dist.init_process_group(
        backend="gloo",
        init_method=f"file://{rendezvous_path}",
        rank=rank,
        world_size=TP_SIZE,
    )
    try:
        _check_mlp(rank)
        _check_attention(rank)
        dist.barrier()
        if rank == 0:
            print(f"TP1 and TP{TP_SIZE} outputs and gradients match.", flush=True)
    finally:
        dist.destroy_process_group()


def main():
    # Frontier login nodes may not resolve their own hostname for Gloo.
    os.environ.setdefault("GLOO_SOCKET_IFNAME", "lo")
    descriptor, rendezvous_path = tempfile.mkstemp(prefix="ucf-vit-tp-")
    os.close(descriptor)
    os.unlink(rendezvous_path)
    try:
        mp.spawn(_worker, args=(rendezvous_path,), nprocs=TP_SIZE, join=True)
    finally:
        if os.path.exists(rendezvous_path):
            os.unlink(rendezvous_path)


if __name__ == "__main__":
    main()
