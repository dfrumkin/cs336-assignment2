import math

import einx
import torch
import triton
import triton.language as tl
from jaxtyping import Float
from torch import Tensor

B_q = 64
B_k = 64


@triton.jit
def flash_fwd_kernel(
    Q_ptr,
    K_ptr,
    V_ptr,
    O_ptr,
    L_ptr,
    stride_qb,
    stride_qq,
    stride_qd,
    stride_kb,
    stride_kk,
    stride_kd,
    stride_vb,
    stride_vk,
    stride_vd,
    stride_ob,
    stride_oq,
    stride_od,
    stride_lb,
    stride_lq,
    N_QUERIES,
    N_KEYS,
    scale: tl.constexpr,
    D: tl.constexpr,
    Q_TILE_SIZE: tl.constexpr,
    K_TILE_SIZE: tl.constexpr,
    is_causal: tl.constexpr,
):
    # Program indices
    query_tile_index = tl.program_id(0)
    batch_index = tl.program_id(1)

    # Offset each pointer with the corresponding batch index
    # multiplied with the batch stride for each tensor
    Q_block_ptr = tl.make_block_ptr(
        Q_ptr + batch_index * stride_qb,
        shape=(N_QUERIES, D),
        strides=(stride_qq, stride_qd),
        offsets=(query_tile_index * Q_TILE_SIZE, 0),
        block_shape=(Q_TILE_SIZE, D),
        order=(1, 0),
    )

    K_block_ptr = tl.make_block_ptr(
        K_ptr + batch_index * stride_kb,
        shape=(N_KEYS, D),
        strides=(stride_kk, stride_kd),
        offsets=(0, 0),
        block_shape=(K_TILE_SIZE, D),
        order=(1, 0),
    )

    V_block_ptr = tl.make_block_ptr(
        V_ptr + batch_index * stride_vb,
        shape=(N_KEYS, D),
        strides=(stride_vk, stride_vd),
        offsets=(0, 0),
        block_shape=(K_TILE_SIZE, D),
        order=(1, 0),
    )

    O_block_ptr = tl.make_block_ptr(
        O_ptr + batch_index * stride_ob,
        shape=(N_QUERIES, D),
        strides=(stride_oq, stride_od),
        offsets=(query_tile_index * Q_TILE_SIZE, 0),
        block_shape=(Q_TILE_SIZE, D),
        order=(1, 0),
    )

    L_block_ptr = tl.make_block_ptr(
        L_ptr + batch_index * stride_lb,
        shape=(N_QUERIES,),
        strides=(stride_lq,),
        offsets=(query_tile_index * Q_TILE_SIZE,),
        block_shape=(Q_TILE_SIZE,),
        order=(0,),
    )

    # Load the query tile
    Q_tile = tl.load(Q_block_ptr, boundary_check=(0, 1), padding_option="zero")

    # Output, logsumexp, maximum for the tile - all in float32 (default)
    o_i = tl.zeros((Q_TILE_SIZE, D))
    l_i = tl.zeros((Q_TILE_SIZE,))
    m_i = tl.full((Q_TILE_SIZE,), -float("inf"))

    # Indices for causal masking
    if is_causal:
        q_indices = query_tile_index * Q_TILE_SIZE + tl.arange(0, Q_TILE_SIZE)

    for key_tile_index in range(tl.cdiv(N_KEYS, K_TILE_SIZE)):
        # Key and value tiles
        K_tile = tl.load(K_block_ptr, boundary_check=(0, 1), padding_option="zero")
        V_tile = tl.load(V_block_ptr, boundary_check=(0, 1), padding_option="zero")

        # Logits from queries and keys
        s_ij = tl.dot(Q_tile, tl.trans(K_tile)) * scale

        # Apply the mask
        if is_causal:
            k_indices = key_tile_index * K_TILE_SIZE + tl.arange(0, K_TILE_SIZE)
            mask = q_indices[:, None] < k_indices[None, :]
            s_ij = tl.where(mask, -1e6, s_ij)

        # The new maximum and the associated correction factor
        m_ij = tl.maximum(m_i, tl.max(s_ij, axis=1))
        correction_factor = tl.exp(m_i - m_ij)
        m_i = m_ij

        # Scores
        p_ij = tl.exp(s_ij - m_i[:, None])

        # Denominator
        l_i = correction_factor * l_i + tl.sum(p_ij, axis=1)

        # Numerator
        o_i = correction_factor[:, None] * o_i + tl.dot(p_ij.to(V_tile.dtype), V_tile)

        # Advance key and value block pointers
        K_block_ptr = K_block_ptr.advance((K_TILE_SIZE, 0))
        V_block_ptr = V_block_ptr.advance((K_TILE_SIZE, 0))

    # Compute the output and logsumexp for the i'th query tile
    o_i = (o_i / l_i[:, None]).to(O_block_ptr.dtype)
    l_i = m_i + tl.log(l_i)

    # Write outputs and logsumexp for the query tile
    tl.store(O_block_ptr, o_i, boundary_check=(0, 1))
    tl.store(L_block_ptr, l_i, boundary_check=(0,))


class FlashTriton(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx: torch.autograd.function.FunctionCtx,
        q: Float[Tensor, " ... n_q d"],
        k: Float[Tensor, " ... n_k d"],
        v: Float[Tensor, " ... n_k d"],
        is_causal: bool = False,
    ) -> Float[Tensor, " ... n_q d"]:
        """FlashAttention 2 forward pass in triton

        Args:
            ctx (torch.autograd.function.FunctionCtx): Autograd context
            q (Float[Tensor, " ... n_q d"]): Query
            k (Float[Tensor, " ... n_k d"]): Key
            v (Float[Tensor, " ... n_k d"]): Value
            is_causal (bool, optional): Whether to apply causal masking.  Defaults to False.
        Returns:
            Float[Tensor, " ... n_q d"]: Output tensor
        """

        # Flatten batch-like dimensions
        qf: Float[Tensor, " b n_q d"] = einx.rearrange("... n_q d -> (...) n_q d", q)  # type: ignore
        kf: Float[Tensor, " b n_k d"] = einx.rearrange("... n_k d -> (...) n_k d", k)  # type: ignore
        vf: Float[Tensor, " b n_k d"] = einx.rearrange("... n_k d -> (...) n_k d", v)  # type: ignore

        # Input dimensions
        batch, n_q, d_model = qf.shape
        n_k = kf.shape[-2]

        assert q.is_cuda and k.is_cuda and v.is_cuda, "Expected CUDA tensors"
        assert q.is_contiguous() and k.is_contiguous() and v.is_contiguous(), "Expect contiguous tensors"
        if is_causal:
            assert n_q == n_k, "Expect equal sequence (query, key, value) lengths when is_causal is True"

        # Prepare the outputs
        o = torch.empty_like(q)
        logsumexp = torch.empty(q.shape[:-1], dtype=torch.float32, device=q.device)
        # These are contiguous views into contiguous tensors
        # Slightly easier this way than trying to restore the original o and logsumexp after Triton
        of: Float[Tensor, " b n_q d"] = einx.rearrange("... n_q d -> (...) n_q d", o)  # type: ignore
        lf: Float[Tensor, " b n_q"] = einx.rearrange("... n_q -> (...) n_q", logsumexp)  # type: ignore

        scale = 1.0 / math.sqrt(d_model)

        flash_fwd_kernel[(triton.cdiv(n_q, B_q), batch)](  # type: ignore
            qf,
            kf,
            vf,
            of,
            lf,
            *qf.stride(),
            *kf.stride(),
            *vf.stride(),
            *of.stride(),
            *lf.stride(),
            N_QUERIES=n_q,
            N_KEYS=n_k,
            scale=scale,
            D=d_model,
            Q_TILE_SIZE=B_q,
            K_TILE_SIZE=B_k,
            is_causal=is_causal,
        )

        # Save data for the backward pass
        ctx.save_for_backward(q, k, v, o, logsumexp)  # TODO: Do I need to save in the original shapes?
        ctx.is_causal = is_causal  # type: ignore

        # Return the output
        return o

    @staticmethod
    @torch.compile(fullgraph=True)
    def _backward_impl(
        d_o: Float[Tensor, " ... n_q d"],
        q: Float[Tensor, " ... n_q d"],
        k: Float[Tensor, " ... n_k d"],
        v: Float[Tensor, " ... n_k d"],
        o: Float[Tensor, " ... n_q d"],
        logsumexp: Float[Tensor, " ... n_q"],
        is_causal: bool,
    ):
        n_q, d_model = q.shape[-2:]
        n_k = k.shape[-2]
        scale = 1 / math.sqrt(d_model)

        d = einx.sum(" ... n_q d -> ... n_q", o * d_o)

        # Recompute logits from queries and keys
        s = einx.dot("... n_q d, ...  n_k d -> ... n_q n_k", q, k, n_q=n_q, n_k=n_k, d=d_model) * scale

        # Apply the causal mask
        if is_causal:
            mask = torch.triu(torch.ones(n_q, n_q, dtype=torch.bool, device=s.device), diagonal=1)
            s = s.masked_fill(mask, -torch.inf)

        # Recompute scores
        p = torch.exp(einx.subtract("... n_q n_k, ... n_q -> ... n_q n_k", s, logsumexp, n_q=n_q, n_k=n_k))

        # Compute gradients
        d_v = einx.dot("... n_q n_k, ... n_q d -> ... n_k d", p, d_o, n_q=n_q, n_k=n_k, d=d_model)
        d_p = einx.dot("... n_q d, ... n_k d -> ... n_q n_k", d_o, v, n_q=n_q, n_k=n_k, d=d_model)
        d_s = einx.multiply(
            "... n_q n_k, ... n_q n_k -> ... n_q n_k", p, einx.subtract("... n_q n_k, ... n_q -> ... n_q n_k", d_p, d)
        )
        d_q = einx.dot("... n_q n_k, ... n_k d -> ... n_q d", d_s, k, n_q=n_q, n_k=n_k, d=d_model) * scale
        d_k = einx.dot("... n_q n_k, ... n_q d -> ... n_k d", d_s, q, n_q=n_q, n_k=n_k, d=d_model) * scale

        # Output gradients
        return d_q, d_k, d_v

    @staticmethod
    def backward(ctx, d_o: torch.Tensor):
        """FlashAttention 2 backward pass in pytorch

        Args:
            ctx (torch.autograd.function.FunctionCtx): Autograd context
            d_o (Float[Tensor, "... n_q d"]): Output gradient

        Returns:
            Gradients for the inputs to forward: dQ, dK, dV, None (for is_causal)
        """
        # Get saved data
        q, k, v, o, logsumexp = ctx.saved_tensors  # type: ignore
        is_causal = ctx.is_causal

        # Call the compiled helper
        d_q, d_k, d_v = FlashTriton._backward_impl(d_o, q, k, v, o, logsumexp, is_causal)
        return d_q, d_k, d_v, None
