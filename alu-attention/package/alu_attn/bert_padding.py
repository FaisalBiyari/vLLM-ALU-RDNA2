"""bert_padding — compatibility with `flash_attn.bert_padding`.

Many projects that simply import flash_attn also use unpad_input/pad_input from this module to
pack padded batches into varlen layout. Without it, drop-in compatibility fails at import time.

Implemented in pure torch without einops: rearrange operations are expressed with reshape.
Gradients are preserved through the same indexed autograd structure used by upstream.
"""
import torch
import torch.nn.functional as F

__all__ = ["index_first_axis", "index_put_first_axis", "index_first_axis_residual",
           "pad_input", "unpad_input", "unpad_input_for_concatenated_sequences",
           "IndexFirstAxis", "IndexPutFirstAxis", "IndexFirstAxisResidual"]


class IndexFirstAxis(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input, indices):
        ctx.save_for_backward(indices)
        assert input.ndim >= 2
        ctx.first_axis_dim, other_shape = input.shape[0], input.shape[1:]
        second_dim = other_shape.numel()
        return torch.gather(
            input.reshape(ctx.first_axis_dim, second_dim), 0,
            indices.unsqueeze(-1).expand(indices.shape[0], second_dim),
        ).reshape(-1, *other_shape)

    @staticmethod
    def backward(ctx, grad_output):
        (indices,) = ctx.saved_tensors
        assert grad_output.ndim >= 2
        other_shape = grad_output.shape[1:]
        grad_output = grad_output.reshape(grad_output.shape[0], other_shape.numel())
        grad_input = torch.zeros([ctx.first_axis_dim, grad_output.shape[1]],
                                 device=grad_output.device, dtype=grad_output.dtype)
        grad_input.scatter_(0, indices.unsqueeze(-1).expand(indices.shape[0], grad_output.shape[1]),
                            grad_output)
        return grad_input.reshape(ctx.first_axis_dim, *other_shape), None


index_first_axis = IndexFirstAxis.apply


class IndexPutFirstAxis(torch.autograd.Function):
    @staticmethod
    def forward(ctx, values, indices, first_axis_dim):
        ctx.save_for_backward(indices)
        assert indices.ndim == 1
        assert values.ndim >= 2
        output = torch.zeros(first_axis_dim, *values.shape[1:], device=values.device,
                             dtype=values.dtype)
        output[indices] = values
        return output

    @staticmethod
    def backward(ctx, grad_output):
        (indices,) = ctx.saved_tensors
        return grad_output[indices], None, None


index_put_first_axis = IndexPutFirstAxis.apply


class IndexFirstAxisResidual(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input, indices):
        ctx.save_for_backward(indices)
        assert input.ndim >= 2
        ctx.first_axis_dim, other_shape = input.shape[0], input.shape[1:]
        output = input[indices]
        # Return input unchanged for the residual path; the caller combines it with the output.
        return output, input.detach()

    @staticmethod
    def backward(ctx, grad_output, grad_residual):
        (indices,) = ctx.saved_tensors
        assert grad_output.ndim >= 2
        grad_input = grad_residual
        indices = indices.reshape(indices.shape[0], *((1,) * (grad_output.ndim - 1)))
        indices = indices.expand_as(grad_output)
        grad_input.scatter_add_(0, indices, grad_output)
        return grad_input, None


index_first_axis_residual = IndexFirstAxisResidual.apply


def unpad_input(hidden_states, attention_mask, unused_mask=None):
    """hidden_states: (batch, seqlen, ...); attention_mask: (batch, seqlen), 1 for real tokens.

    Returns the same five values as flash-attn:
    (hidden_states_unpad, indices, cu_seqlens, max_seqlen_in_batch, used_seqlens_in_batch).
    """
    all_masks = (attention_mask + unused_mask) if unused_mask is not None else attention_mask
    seqlens_in_batch = all_masks.sum(dim=-1, dtype=torch.int32)
    used_seqlens_in_batch = attention_mask.sum(dim=-1, dtype=torch.int32)
    indices = torch.nonzero(all_masks.flatten(), as_tuple=False).flatten()
    max_seqlen_in_batch = int(seqlens_in_batch.max().item())
    cu_seqlens = F.pad(torch.cumsum(seqlens_in_batch, dim=0, dtype=torch.int32), (1, 0))
    flat = hidden_states.reshape(-1, *hidden_states.shape[2:])
    return (index_first_axis(flat, indices), indices, cu_seqlens, max_seqlen_in_batch,
            used_seqlens_in_batch)


def unpad_input_for_concatenated_sequences(hidden_states, attention_mask_in_length):
    """attention_mask_in_length: (batch, seqlen), containing concatenated-sequence LENGTHS per row
    (zeros are padding). Returns the same four values as flash-attn."""
    length = attention_mask_in_length.sum(dim=-1)
    seqlen = attention_mask_in_length.size(-1)
    attention_mask_2d = (torch.arange(seqlen, device=length.device).unsqueeze(0)
                         < length.unsqueeze(-1))
    real_indices_idx = torch.nonzero(attention_mask_in_length.flatten(), as_tuple=False).flatten()
    seqlens_in_batch = attention_mask_in_length.flatten()[real_indices_idx]
    indices = torch.nonzero(attention_mask_2d.flatten(), as_tuple=False).flatten()
    max_seqlen_in_batch = int(seqlens_in_batch.max().item())
    cu_seqlens = F.pad(torch.cumsum(seqlens_in_batch, dim=0, dtype=torch.int32), (1, 0))
    flat = hidden_states.reshape(-1, *hidden_states.shape[2:])
    return index_first_axis(flat, indices), indices, cu_seqlens, max_seqlen_in_batch


def pad_input(hidden_states, indices, batch, seqlen):
    """Convert varlen data back to (batch, seqlen, ...)."""
    output = index_put_first_axis(hidden_states, indices, batch * seqlen)
    return output.reshape(batch, seqlen, *output.shape[1:])
