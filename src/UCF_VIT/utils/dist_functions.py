import torch
import torch.distributed as dist
from torch.autograd import Function
# The two imports below are not always available depending on the
# USE_DISTRIBUTED compile flag. Make sure they raise import error
# if we're trying to use them.
from torch.distributed import group, ReduceOp

def broadcast(tensor, src, group=group.WORLD):
    """
    Broadcasts the tensor to the whole group.

    ``tensor`` must have the same number of elements in all processes
    participating in the collective.

    Arguments:
        tensor (Tensor): Data to be sent if ``src`` is the rank of current
            process.
        src (int): Source rank.
        group (ProcessGroup, optional): The process group to work on.

    Returns:
        Tensor: Received tensor from the broadcast op.

    """
    return _Broadcast.apply(src, group, tensor)




def F_Broadcast_B_Identity(tensor, src, group=group.WORLD):
    """
    Broadcasts the tensor to the whole group.

    ``tensor`` must have the same number of elements in all processes
    participating in the collective.

    Arguments:
        tensor (Tensor): Data to be sent if ``src`` is the rank of current
            process.
        src (int): Source rank.
        group (ProcessGroup, optional): The process group to work on.

    Returns:
        Tensor: Received tensor from the broadcast op.

    """
    return _F_Broadcast_B_Identity.apply(src, group, tensor)




def F_Identity_B_AllReduce(tensor, group=group.WORLD):
    """
    Broadcasts the tensor to the whole group.

    ``tensor`` must have the same number of elements in all processes
    participating in the collective.

    Arguments:
        tensor (Tensor): Data to be sent if ``src`` is the rank of current
            process.
        group (ProcessGroup, optional): The process group to work on.

    Returns:
        Tensor: Received tensor from the broadcast op.

    """
    return _F_Identity_B_AllReduce.apply(group, tensor)



def F_Identity_B_AllReduce_VariableMapping(tensor, group=group.WORLD):
    """
    Broadcasts the tensor to the whole group.

    ``tensor`` must have the same number of elements in all processes
    participating in the collective.

    Arguments:
        tensor (Tensor): Data to be sent if ``src`` is the rank of current
            process.
        group (ProcessGroup, optional): The process group to work on.

    Returns:
        Tensor: Received tensor from the broadcast op.

    """
    return _F_Identity_B_AllReduce_VariableMapping.apply(group, tensor)




def gather(tensor, dst=0, group=group.WORLD):
    """
    Gathers a list of tensors in a single process.

    Arguments:
        tensor (Tensor): Input tensor.
        dst (int, optional): Destination rank (default is 0).
        group (ProcessGroup, optional): The process group to work on.

    Returns:
        tuple[Tensor]: List of appropriately-sized tensors with the gathered data.
    """
    return _Gather.apply(dst, group, tensor)


def scatter(tensors, src=0, group=group.WORLD):
    """
    Scatters a list of tensors to all processes in a group.

    Each process will receive exactly one tensor and store its data in the
    ``tensor`` argument.

    Arguments:
        tensors (list[Tensor]): List of tensors to scatter on the source rank.
            Receivers must pass ``None`.
        src (int, optional): Source rank (default is 0).
        group (ProcessGroup, optional): The process group to work on.

    Returns:
        Tensor: Output tensor from the scatter operation.

    """
    return _Scatter.apply(src, group, *tensors)


def F_AllReduce_B_Identity(tensor, op=ReduceOp.SUM, group=group.WORLD):
    """
    Reduces the tensor data across all machines.

    Only the process with rank ``dst`` is going to receive the final result.

    Arguments:
        tensor (Tensor): Input of the collective.
        op (optional): One of the values from
            ``torch.distributed.ReduceOp``
            enum.  Specifies an operation used for element-wise reductions.
        group (ProcessGroup, optional): The process group to work on.

    Returns:
        Tensor: Output of the collective.

    """
    return _F_AllReduce_B_Identity.apply(op, group, tensor)


def F_AllReduce_B_Identity_VariableMapping(tensor, op=ReduceOp.SUM, group=group.WORLD):
    """
    Reduces the tensor data across all machines.

    Only the process with rank ``dst`` is going to receive the final result.

    Arguments:
        tensor (Tensor): Input of the collective.
        op (optional): One of the values from
            ``torch.distributed.ReduceOp``
            enum.  Specifies an operation used for element-wise reductions.
        group (ProcessGroup, optional): The process group to work on.

    Returns:
        Tensor: Output of the collective.

    """
    return _F_AllReduce_B_Identity_VariableMapping.apply(op, group, tensor)





def F_Identity_B_Broadcast(tensor, src, group=group.WORLD):
    """
    broadcast the tensor gradient across all machines.

    Only the process with rank ``src`` is going to receive the final result.

    Arguments:
        tensor (Tensor): Input of the collective.
        src (int): source rank.
        op (optional): One of the values from
            ``torch.distributed.ReduceOp``
            enum.  Specifies an operation used for element-wise reductions.
        group (ProcessGroup, optional): The process group to work on.

    Returns:
        Tensor: Output of the collective.

    """
    return _F_Identity_B_Broadcast.apply(src, group, tensor)




def reduce(tensor, dst, op=ReduceOp.SUM, group=group.WORLD):
    """
    Reduces the tensor data across all machines.

    Only the process with rank ``dst`` is going to receive the final result.

    Arguments:
        tensor (Tensor): Input of the collective.
        dst (int): Destination rank.
        op (optional): One of the values from
            ``torch.distributed.ReduceOp``
            enum.  Specifies an operation used for element-wise reductions.
        group (ProcessGroup, optional): The process group to work on.

    Returns:
        Tensor: Output of the collective.

    """
    return _Reduce.apply(dst, op, group, tensor)


def reduce_scatter(output, input_list, op=ReduceOp.SUM, group=group.WORLD):
    """
    Reduces, then scatters a list of tensors to all processes in a group.

    Arguments:
        output (Tensor): Output tensor.
        input_list (list[Tensor]): List of tensors to reduce and scatter.
        op (optional): One of the values from
            ``torch.distributed.ReduceOp``
            enum.  Specifies an operation used for element-wise reductions.
        group (ProcessGroup, optional): The process group to work on.

    Returns:
        Tensor: Output of the collective.

    """
    return _Reduce_Scatter.apply(op, group, output, *input_list)


def all_gather(tensor, group=group.WORLD):
    """
    Gathers tensors from the whole group in a list.

    Arguments:
        tensor (Tensor): Tensor to be broadcast from current process.
        group (ProcessGroup, optional): The process group to work on.

    Returns:
        tuple([Tensor]): Output of the collective.

    """
    return _AllGather.apply(group, tensor)

def _all_gather_base(output_tensor, input_tensor, group=group.WORLD):
    """
    Single tensor all gather. Gathers a single tensor from all ranks, and puts them in a single output tensor.

    Args:
        output_tensor (Tensor): Output tensor. It should contain
            correctly-sized tensors to be used for output of the collective.
        input_tensor (Tensor): Tensor to be broadcast from current process.
        group (ProcessGroup, optional): The process group to work on. If None,
            the default process group will be used.
        async_op (bool, optional): Whether this op should be an async op

    Returns:
        Async work handle, if async_op is set to True.
        None, if not async_op or if not part of the group

    Examples:
        >>> # All tensors below are of torch.int64 dtype.
        >>> # We have 2 process groups, 2 ranks.
        >>> # xdoctest: +SKIP("incorrect want text")
        >>> output_tensor = torch.zeros(2, dtype=torch.int64)
        >>> output_tensor
        [tensor([0, 0])] # Rank 0 and 1
        >>> tensor = torch.arange(1, dtype=torch.int64) + 1 + rank
        >>> tensor
        tensor([1]) # Rank 0
        tensor([2]) # Rank 1
        >>> dist.all_gather_base(output_tensor, tensor)
        >>> output_tensor
        tensor([1,2]) # Rank 0
        tensor([1,2]) # Rank 1

    .. warning::
        `_all_gather_base` is experimental and subject to change.
        It is the caller's responsibility to ensure the output_tensor
        is correctly sized.

    """
    return _AllGatherBase.apply(output_tensor, input_tensor, group)


def all_to_all(output_tensor_list, input_tensor_list, group=group.WORLD):
    """
    Each process scatters list of input tensors to all processes in a group and
    return gathered list of tensors in output list.

    Arguments:
        out_tensor_list (list[Tensor]): list of tensors to gather one per rank.
        input_tensor_list (list[Tensor]): List of tensors to scatter one per rank.
        group (ProcessGroup, optional): The process group to work on.

    Returns:
        tuple([Tensor]): Output of the collective.

    """
    return _AlltoAll.apply(group, output_tensor_list, *input_tensor_list)


def all_to_all_single(
    output,
    input,
    output_split_sizes=None,
    input_split_sizes=None,
    group=group.WORLD,
):
    """
    Each process splits input tensor and then scatters the split list
    to all processes in a group. Then concatenate the received tensors from all
    the processes in the group and return single output tensor.

    Arguments:
        output (Tensor): Gathered cancatenated output tensor.
        input (Tensor): Input tensor to scatter.
        output_split_sizes: (list[Int], optional): Output split sizes for dim 0
            if specified None or empty, dim 0 of ``output`` tensor must divide
            equally by ``world_size``.
        input_split_sizes: (list[Int], optional): Input split sizes for dim 0
            if specified None or empty, dim 0 of ``input`` tensor must divide
            equally by ``world_size``.

    Returns:
        Tensor: Output of the collective.

    """
    return _AlltoAllSingle.apply(
        group, output, output_split_sizes, input_split_sizes, input
    )


def all_reduce(tensor, op=ReduceOp.SUM, group=group.WORLD):
    """
    Reduces the tensor data across all machines in such a way that all get
    the final result.

    After the call the returned tensor is going to be bitwise
    identical in all processes.

    Arguments:
        tensor (Tensor): Input of the collective.
        op (optional): One of the values from
            ``torch.distributed.ReduceOp``
            enum.  Specifies an operation used for element-wise reductions.
        group (ProcessGroup, optional): The process group to work on.

    Returns:
        Tensor: Output of the collective

    """
    return _AllReduce.apply(op, group, tensor)




class _F_Broadcast_B_Identity(Function):
    """Autograd function that broadcasts on the forward pass and is an identity on the backward pass."""

    @staticmethod
    def forward(ctx, src, group, tensor):
        """Broadcasts `tensor` from rank `src` to every rank in `group`.

        Args:
            ctx: Autograd context.
            src: Source rank to broadcast from.
            group: Process group to broadcast within.
            tensor: Tensor to broadcast (cloned before the in-place collective).

        Returns:
            The broadcast tensor.
        """
        ctx.src = src
        ctx.group = group
        ctx.rank = dist.get_rank()
        # torch.distributed makes all the calls in place
        # we allocate new tensors to avoid this
        tensor = tensor.clone()
        dist.broadcast(tensor, src, group=group)
        return tensor

    @staticmethod
    def backward(ctx, grad_output):
        """Passes the incoming gradient through unchanged on every rank.

        Args:
            ctx: Autograd context (unused).
            grad_output: Gradient with respect to the forward output.

        Returns:
            A tuple of gradients matching `forward`'s inputs: `(None, None, grad_output.clone())`.
        """

        return (None, None, grad_output.clone())





class _Broadcast(Function):
    """Autograd function that broadcasts on the forward pass and reduce-sums gradients back to `src` on the backward pass."""

    @staticmethod
    def forward(ctx, src, group, tensor):
        """Broadcasts `tensor` from rank `src` to every rank in `group`, in place.

        Args:
            ctx: Autograd context.
            src: Source rank to broadcast from.
            group: Process group to broadcast within.
            tensor: Tensor to broadcast.

        Returns:
            The broadcast tensor.
        """
        ctx.src = src
        ctx.group = group
        ctx.rank = dist.get_rank()
        # torch.distributed makes all the calls in place
        # we allocate new tensors to avoid this
        #tensor = tensor.clone()
        dist.broadcast(tensor, src, group=group)
        return tensor

    @staticmethod
    def backward(ctx, grad_output):
        """Reduce-sums incoming gradients back to rank `src`, zeroing them elsewhere.

        Args:
            ctx: Autograd context holding `src`, `group`, and `rank`.
            grad_output: Gradient with respect to the forward output.

        Returns:
            A tuple of gradients matching `forward`'s inputs: `(None, None, gx)`,
            where `gx` is nonzero only on rank `src`.
        """
        gx = _Reduce.apply(ctx.src, ReduceOp.SUM, ctx.group, grad_output)
        if ctx.src != ctx.rank:
            gx.zero_()
        return (None, None, gx)


class _F_Identity_B_AllReduce(Function):
    """Autograd function that is an identity on the forward pass and all-reduce-sums gradients on the backward pass."""

    @staticmethod
    def forward(ctx, group, tensor):
        """Returns `tensor` unchanged.

        Args:
            ctx: Autograd context.
            group: Process group to use for the backward all-reduce.
            tensor: Input tensor.

        Returns:
            `tensor`, unchanged.
        """
        ctx.group = group
        # torch.distributed makes all the calls in place
        # we allocate new tensors to avoid this
        #tensor = tensor.clone()
        #dist.broadcast(tensor, src, group=group)
        return tensor

    @staticmethod
    def backward(ctx, grad_output):
        """All-reduce-sums the incoming gradient across `ctx.group`.

        Args:
            ctx: Autograd context holding `group`.
            grad_output: Gradient with respect to the forward output.

        Returns:
            A tuple of gradients matching `forward`'s inputs: `(None, gx)`.
        """

        gx = _AllReduce.apply(ReduceOp.SUM, ctx.group, grad_output)
        return (None, gx)



class _F_Identity_B_AllReduce_VariableMapping(Function):
    """Variable-mapping variant of `_F_Identity_B_AllReduce`: identity forward, all-reduce-sum backward."""

    @staticmethod
    def forward(ctx, group, tensor):
        """Returns `tensor` unchanged.

        Args:
            ctx: Autograd context.
            group: Process group to use for the backward all-reduce.
            tensor: Input tensor.

        Returns:
            `tensor`, unchanged.
        """
        ctx.group = group
        # torch.distributed makes all the calls in place
        # we allocate new tensors to avoid thiis
        #tensor = tensor.clone()

#        print("rank",dist.get_rank(),"F_Identity tensor shape",tensor.shape,"tensor[0,0,0]",tensor[0,0,0],flush=True)
        #dist.broadcast(tensor, src, group=group)
        return tensor

    @staticmethod
    def backward(ctx, grad_output):
        """All-reduce-sums the incoming gradient across `ctx.group`.

        Args:
            ctx: Autograd context holding `group`.
            grad_output: Gradient with respect to the forward output.

        Returns:
            A tuple of gradients matching `forward`'s inputs: `(None, gx)`.
        """

        gx = _AllReduce.apply(ReduceOp.SUM, ctx.group, grad_output)
        return (None, gx)




class _Gather(Function):
    """Autograd function that gathers tensors onto `dst` on the forward pass and scatters gradients back on the backward pass."""

    @staticmethod
    def forward(ctx, dst, group, tensor):
        """Gathers `tensor` from every rank in `group` onto rank `dst`.

        Args:
            ctx: Autograd context.
            dst: Destination rank.
            group: Process group to gather within.
            tensor: This rank's contribution to gather; must be correctly sized.

        Returns:
            Tuple of gathered tensors, one per rank in `group`.
        """
        ctx.dst = dst
        ctx.group = group
        # Need to create a list of tensors here to do the
        # aggregation, get it from the group size
        # tensor should be correctly sized for the method
        # gathering
        tensor_list = [
            torch.zeros_like(tensor) for i in range(dist.get_world_size(group=group))
        ]

        tensor = tensor.contiguous()
        if dist.get_rank(group=group) == dst:
            dist.gather(tensor, tensor_list, dst, group=group)
        else:
            dist.gather(tensor, None, dst, group=group)
        return tuple(tensor_list)

    @staticmethod
    def backward(ctx, *grad_outputs):
        """Scatters the incoming gradients from `dst` back to each rank.

        Args:
            ctx: Autograd context holding `dst` and `group`.
            *grad_outputs: Gradients with respect to each gathered tensor.

        Returns:
            A tuple of gradients matching `forward`'s inputs: `(None, None, gx)`.
        """
        return (None, None) + (_Scatter.apply(ctx.dst, ctx.group, *grad_outputs),)


class _Scatter(Function):
    """Autograd function that scatters tensors from `src` on the forward pass and gathers gradients back on the backward pass."""

    @staticmethod
    def forward(ctx, src, group, *tensors):
        """Scatters one tensor per rank in `group` from rank `src`.

        Args:
            ctx: Autograd context.
            src: Source rank holding the tensors to scatter.
            group: Process group to scatter within.
            *tensors: On rank `src`, the list of tensors to scatter (all the same
                size), one per rank in `group`; ignored on other ranks.

        Returns:
            The tensor received by the current rank.
        """
        ctx.src = src
        ctx.group = group
        assert all(t.size() == tensors[0].size() for t in tensors)
        output = torch.zeros_like(tensors[0])
        if dist.get_rank(group=group) == src:
            dist.scatter(output, list(tensors), src, group=group)
        else:
            dist.scatter(output, None, src, group=group)
        return output

    @staticmethod
    def backward(ctx, grad_output):
        """Gathers the incoming per-rank gradients back onto rank `src`.

        Args:
            ctx: Autograd context holding `src` and `group`.
            grad_output: Gradient with respect to the forward output on this rank.

        Returns:
            A tuple of gradients matching `forward`'s inputs: `(None, None,
            *grad_tensors)`.
        """
        return (None, None) + _Gather.apply(ctx.src, ctx.group, grad_output)


class _Reduce(Function):
    """Autograd function that reduces to `src` on the forward pass and broadcasts gradients from `src` on the backward pass."""

    @staticmethod
    def forward(ctx, src, op, group, tensor):
        """Reduces `tensor` across `group` onto rank `src` using reduction `op`.

        Args:
            ctx: Autograd context.
            src: Destination rank that receives the reduced result.
            op: Reduction op from `torch.distributed.ReduceOp`.
            group: Process group to reduce within.
            tensor: Input tensor (cloned before the in-place collective).

        Returns:
            The reduced tensor (only meaningful on rank `src`).
        """
        ctx.src = src
        ctx.group = group
        tensor = tensor.clone()
        dist.reduce(tensor, src, op=op, group=group)
        return tensor

    @staticmethod
    def backward(ctx, grad_output):
        """Broadcasts the gradient from rank `src` to every rank in `group`.

        Args:
            ctx: Autograd context holding `src` and `group`.
            grad_output: Gradient with respect to the forward output.

        Returns:
            A tuple of gradients matching `forward`'s inputs: `(None, None, None, gx)`.
        """
        return (None, None, None) + (_Broadcast.apply(ctx.src, ctx.group, grad_output),)




class _F_Identity_B_Broadcast(Function):
    """Autograd function that is an identity on the forward pass and broadcasts gradients from `src` on the backward pass."""

    @staticmethod
    def forward(ctx, src, group, tensor):
        """Returns `tensor` unchanged.

        Args:
            ctx: Autograd context.
            src: Rank whose gradient is broadcast on the backward pass.
            group: Process group to use for the backward broadcast.
            tensor: Input tensor.

        Returns:
            `tensor`, unchanged.
        """
        ctx.src = src
        ctx.group = group
        #tensor = tensor.clone()
        return tensor

    @staticmethod
    def backward(ctx, grad_output):
        """Broadcasts the gradient on rank `src` to every rank in `group`.

        Args:
            ctx: Autograd context holding `src` and `group`.
            grad_output: Gradient with respect to the forward output.

        Returns:
            A tuple of gradients matching `forward`'s inputs: `(None, None, gx)`.
        """
        return (None,  None) + (_Broadcast.apply(ctx.src, ctx.group, grad_output.contiguous()),)



class _F_AllReduce_B_Identity(Function):
    """Autograd function that all-reduces on the forward pass and is an identity on the backward pass."""

    @staticmethod
    def forward(ctx, op, group, tensor):
        """All-reduces `tensor` across `group` using reduction `op`.

        Args:
            ctx: Autograd context.
            op: Reduction op from `torch.distributed.ReduceOp`.
            group: Process group to reduce within.
            tensor: Input tensor (cloned before the in-place collective).

        Returns:
            The all-reduced tensor, identical on every rank in `group`.
        """
        ctx.group = group

        tensor = tensor.clone()
        dist.all_reduce(tensor, op=op, group=group)

        return tensor

    @staticmethod
    def backward(ctx, grad_output):
        """Passes the incoming gradient through unchanged.

        Args:
            ctx: Autograd context (unused).
            grad_output: Gradient with respect to the forward output.

        Returns:
            A tuple of gradients matching `forward`'s inputs: `(None, None, grad_output.clone())`.
        """
#        return (None, None, None) + (_Broadcast.apply(ctx.src, ctx.group, grad_output),)
        return (None, None, grad_output.clone())



class _F_AllReduce_B_Identity_VariableMapping(Function):
    """Variable-mapping variant of `_F_AllReduce_B_Identity`: all-reduce forward, identity backward."""

    @staticmethod
    def forward(ctx, op, group, tensor):
        """All-reduces `tensor` across `group` using reduction `op`.

        Args:
            ctx: Autograd context.
            op: Reduction op from `torch.distributed.ReduceOp`.
            group: Process group to reduce within.
            tensor: Input tensor (cloned before the in-place collective).

        Returns:
            The all-reduced tensor, identical on every rank in `group`.
        """
        ctx.group = group

        tensor = tensor.clone()
        dist.all_reduce(tensor, op=op, group=group)

        return tensor

    @staticmethod
    def backward(ctx, grad_output):
        """Passes the incoming gradient through unchanged.

        Args:
            ctx: Autograd context (unused).
            grad_output: Gradient with respect to the forward output.

        Returns:
            A tuple of gradients matching `forward`'s inputs: `(None, None, grad_output.clone())`.
        """
#        return (None, None, None) + (_Broadcast.apply(ctx.src, ctx.group, grad_output),)
#        print("rank",dist.get_rank(),"B_Identity_VariableMapping grad_output[0,0,0]",grad_output[0,0,0],"grad_output[0,0,1]",grad_output[0,0,1],"grad_output[0,0,2]",grad_output[0,0,2],flush=True)

        return (None, None, grad_output.clone())




class _Reduce_Scatter(Function):
    """Autograd function that reduces-then-scatters on the forward pass and all-gathers gradients on the backward pass."""

    @staticmethod
    def forward(ctx, op, group, tensor, *input_tensor_list):
        """Reduces `input_tensor_list` across `group` with `op`, then scatters the result.

        Args:
            ctx: Autograd context.
            op: Reduction op from `torch.distributed.ReduceOp`.
            group: Process group to reduce/scatter within.
            tensor: Output tensor to receive this rank's shard, in place.
            *input_tensor_list: List of tensors to reduce and scatter, one per rank.

        Returns:
            `tensor`, holding this rank's shard of the reduced result.
        """
        ctx.group = group
        input_tensor_list = tuple(t.contiguous() for t in input_tensor_list)
        dist.reduce_scatter(tensor, list(input_tensor_list), op=op, group=group)
        return tensor

    @staticmethod
    def backward(ctx, grad_output):
        """All-gathers the incoming per-shard gradient across `group`.

        Args:
            ctx: Autograd context holding `group`.
            grad_output: Gradient with respect to this rank's shard of the output.

        Returns:
            A tuple of gradients matching `forward`'s inputs: `(None, None, None,
            *grad_tensors)`.
        """
        return (None, None, None) + _AllGather.apply(ctx.group, grad_output)


class _AllGather(Function):
    """Autograd function that all-gathers on the forward pass and reduce-scatters gradients on the backward pass."""

    @staticmethod
    def forward(ctx, group, tensor):
        """Gathers `tensor` from every rank in `group` onto every rank.

        Args:
            ctx: Autograd context.
            group: Process group to gather within.
            tensor: This rank's contribution to gather.

        Returns:
            Tuple of gathered tensors, one per rank in `group`, identical on every
            rank.
        """
        # Need contiguous tensors for collectives.
        tensor = tensor.contiguous()

        ctx.group = group
        out_tensor_list = [
            torch.empty_like(tensor) for _ in range(dist.get_world_size(group=group))
        ]

        dist.all_gather(out_tensor_list, tensor, group=group)
        return tuple(out_tensor_list)

    @staticmethod
    def backward(ctx, *grad_outputs):
        """Reduce-scatters the incoming per-rank gradients back to this rank's shard.

        Uses `reduce_scatter` on NCCL, or an all-to-all plus sum as a fallback on
        backends that don't support reduce-scatter.

        Args:
            ctx: Autograd context holding `group`.
            *grad_outputs: Gradients with respect to each gathered tensor.

        Returns:
            A tuple of gradients matching `forward`'s inputs: `(None, gx)`.
        """
        if dist.get_backend(group=ctx.group) is dist.Backend.NCCL:
            rank = dist.get_rank(group=ctx.group)
            gx = torch.empty_like(grad_outputs[rank])
            _Reduce_Scatter.apply(ReduceOp.SUM, ctx.group, gx, *grad_outputs)
        else:
            # As many backends doesn't support ReduceScatter, we use AlltoAll with .sum()
            # to emulate the ReduceScatter behavior
            tensor_list = [torch.empty_like(tensor) for tensor in grad_outputs]
            gxs = _AlltoAll.apply(ctx.group, tensor_list, *grad_outputs)
            gx = torch.sum(torch.stack(gxs), dim=0)
        return (None, gx)

class _AllGatherBase(Function):
    """Autograd function wrapping `dist._all_gather_base`, with a reduce-scatter backward pass (NCCL only)."""

    @staticmethod
    def forward(ctx, output_tensor, input_tensor, group):
        """Gathers `input_tensor` from every rank in `group` into a single `output_tensor`.

        Args:
            ctx: Autograd context.
            output_tensor: Pre-sized tensor to receive the concatenated gathered
                data, in place.
            input_tensor: This rank's contribution to gather.
            group: Process group to gather within.

        Returns:
            `output_tensor`, holding the concatenated gathered data.
        """
        ctx.group = group
        dist._all_gather_base(output_tensor, input_tensor.contiguous(), group=group)
        return output_tensor

    @staticmethod
    def backward(ctx, grad_output):
        """Reduce-scatters the incoming gradient back to this rank's shard.

        Only supported on the NCCL backend.

        Args:
            ctx: Autograd context holding `group`.
            grad_output: Gradient with respect to `output_tensor`; its first
                dimension must be divisible by the group's world size.

        Returns:
            A tuple of gradients matching `forward`'s inputs: `(None, gx, None)`.

        Raises:
            RuntimeError: If `grad_output`'s first dimension isn't divisible by the
                world size, or the backend isn't NCCL.
        """
        if dist.get_backend(group=ctx.group) is dist.Backend.NCCL:
            world_size = dist.get_world_size(group=ctx.group)
            out_size = list(grad_output.size())
            if out_size[0] % world_size != 0:
                raise RuntimeError(
                    f'Tensor with dimensions: {out_size} does '
                    f'not have first dimension divisible by world_size: {world_size}'
                )
            out_size[0] = out_size[0] // dist.get_world_size(group=ctx.group)
            gx = torch.empty(out_size, device=grad_output.device, dtype=grad_output.dtype)
            dist._reduce_scatter_base(gx, grad_output, ReduceOp.SUM, ctx.group)
        else:
            raise RuntimeError("Backend not supported!")
        return (None, gx, None)

class _AlltoAll(Function):
    """Autograd function that all-to-all exchanges lists of tensors, with a matching backward pass."""

    @staticmethod
    def forward(ctx, group, out_tensor_list, *tensors):
        """Scatters `tensors` (one per rank) to all ranks and gathers each rank's contribution.

        Falls back to per-rank scatter calls on the GLOO backend, which doesn't
        support `all_to_all` directly.

        Args:
            ctx: Autograd context.
            group: Process group to exchange within.
            out_tensor_list: Pre-sized list of tensors to receive the gathered
                data, one per rank, in place.
            *tensors: List of tensors to scatter, one per destination rank.

        Returns:
            Tuple of tensors received from each rank.
        """
        ctx.group = group
        ctx.input_tensor_size_list = [
            tensors[i].size() for i in range(dist.get_world_size(group=group))
        ]
        my_rank = dist.get_rank(group=group)
        tensors = tuple(t.contiguous() for t in tensors)
        # Implement it on means of scatter/gather, send/recv async operations have issues
        if dist.get_backend(group=group) is dist.Backend.GLOO:
            for i in range(dist.get_world_size(group=group)):
                to_send = None
                if i == my_rank:
                    to_send = list(tensors)
                dist.scatter(out_tensor_list[i], to_send, i, group=group)
        else:
            dist.all_to_all(
                out_tensor_list,
                list(tensors),
                group=group,
            )
        return tuple(out_tensor_list)

    @staticmethod
    def backward(ctx, *grad_outputs):
        """Runs another all-to-all exchange on the incoming gradients.

        Args:
            ctx: Autograd context holding `group` and `input_tensor_size_list`.
            *grad_outputs: Gradients with respect to each received tensor.

        Returns:
            A tuple of gradients matching `forward`'s inputs: `(None, None,
            *grad_tensors)`.
        """
        tensor_list = [
            torch.empty(size, device=grad_outputs[0].device, dtype=grad_outputs[0].dtype)
            for size in ctx.input_tensor_size_list
        ]
        return (None, None) + _AlltoAll.apply(ctx.group, tensor_list, *grad_outputs)


class _AlltoAllSingle(Function):
    """Autograd function wrapping `dist.all_to_all_single`, with a matching backward pass."""

    @staticmethod
    def forward(ctx, group, output, output_split_sizes, input_split_sizes, input):
        """Splits `input` and all-to-all exchanges it into `output`.

        Args:
            ctx: Autograd context.
            group: Process group to exchange within.
            output: Pre-sized tensor to receive the concatenated exchanged data, in
                place.
            output_split_sizes: Split sizes for `output`'s dim 0, or None/empty to
                split evenly by world size.
            input_split_sizes: Split sizes for `input`'s dim 0, or None/empty to
                split evenly by world size.
            input: Input tensor to split and scatter.

        Returns:
            `output`, holding the concatenated exchanged data.
        """
        ctx.group = group
        ctx.input_size = input.size()
        ctx.output_split_sizes = input_split_sizes
        ctx.input_split_sizes = output_split_sizes
        dist.all_to_all_single(
            output,
            input,
            output_split_sizes=output_split_sizes,
            input_split_sizes=input_split_sizes,
            group=group,
        )
        return output

    @staticmethod
    def backward(ctx, grad_output):
        """Runs the inverse all-to-all exchange (split sizes swapped) on the incoming gradient.

        Args:
            ctx: Autograd context holding `group`, `input_size`, `output_split_sizes`,
                and `input_split_sizes`.
            grad_output: Gradient with respect to the forward `output`.

        Returns:
            A tuple of gradients matching `forward`'s inputs: `(None, None, None,
            None, gx)`.
        """
        tensor = torch.empty(ctx.input_size, device=grad_output.device, dtype=grad_output.dtype)
        return (None, None, None, None) + (
            _AlltoAllSingle.apply(
                ctx.group,
                tensor,
                ctx.output_split_sizes,
                ctx.input_split_sizes,
                grad_output.contiguous(),
            ),
        )


class _AllReduce(Function):
    """Autograd function that all-reduces on both the forward and backward pass."""

    @staticmethod
    def forward(ctx, op, group, tensor):
        """All-reduces `tensor` across `group` using reduction `op`, in place.

        Args:
            ctx: Autograd context.
            op: Reduction op from `torch.distributed.ReduceOp`.
            group: Process group to reduce within.
            tensor: Input tensor.

        Returns:
            `tensor`, all-reduced in place, identical on every rank in `group`.
        """
        ctx.group = group
        ctx.op = op
        #tensor = tensor.clone()
        dist.all_reduce(tensor, op=op, group=group)
        return tensor

    @staticmethod
    def backward(ctx, grad_output):
        """All-reduces the incoming gradient across `ctx.group` using the same op.

        Args:
            ctx: Autograd context holding `op` and `group`.
            grad_output: Gradient with respect to the forward output.

        Returns:
            A tuple of gradients matching `forward`'s inputs: `(None, None, gx)`.
        """
        return (None, None) + (_AllReduce.apply(ctx.op, ctx.group, grad_output),)
