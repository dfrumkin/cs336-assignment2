import math

import einx
import torch
import triton
import triton.language as tl
from jaxtyping import Float
from torch import Tensor

CONFIGS_FWD = [
    # 64×64 with 4 or 8 warps
    triton.Config({"Q_TILE_SIZE": 64, "K_TILE_SIZE": 64}, num_warps=4),
    triton.Config({"Q_TILE_SIZE": 64, "K_TILE_SIZE": 64}, num_warps=8),
    # 64x128 with 4 or 8 warps
    triton.Config({"Q_TILE_SIZE": 64, "K_TILE_SIZE": 128}, num_warps=4),
    triton.Config({"Q_TILE_SIZE": 64, "K_TILE_SIZE": 128}, num_warps=8),
    # 128×64 with 4 or 8 warps
    triton.Config({"Q_TILE_SIZE": 128, "K_TILE_SIZE": 64}, num_warps=4),
    triton.Config({"Q_TILE_SIZE": 128, "K_TILE_SIZE": 64}, num_warps=8),
    # maybe 128×128 with 4 or 8 warps
    triton.Config({"Q_TILE_SIZE": 128, "K_TILE_SIZE": 128}, num_warps=4),
    triton.Config({"Q_TILE_SIZE": 128, "K_TILE_SIZE": 128}, num_warps=8),
]

CONFIGS_BWD = [
    # 32×32 with 4 or 8 warps
    triton.Config({"Q_TILE_SIZE": 32, "K_TILE_SIZE": 32}, num_warps=4),
    triton.Config({"Q_TILE_SIZE": 32, "K_TILE_SIZE": 32}, num_warps=8),
    # 32×64 with 4 or 8 warps
    triton.Config({"Q_TILE_SIZE": 32, "K_TILE_SIZE": 64}, num_warps=4),
    triton.Config({"Q_TILE_SIZE": 32, "K_TILE_SIZE": 64}, num_warps=8),
    # 64×32 with 4 or 8 warps
    triton.Config({"Q_TILE_SIZE": 64, "K_TILE_SIZE": 32}, num_warps=4),
    triton.Config({"Q_TILE_SIZE": 64, "K_TILE_SIZE": 32}, num_warps=8),
    # 64×64 with 4 or 8 warps
    triton.Config({"Q_TILE_SIZE": 64, "K_TILE_SIZE": 64}, num_warps=4),
    triton.Config({"Q_TILE_SIZE": 64, "K_TILE_SIZE": 64}, num_warps=8),
    # 64x128 with 4 or 8 warps
    triton.Config({"Q_TILE_SIZE": 64, "K_TILE_SIZE": 128}, num_warps=4),
    triton.Config({"Q_TILE_SIZE": 64, "K_TILE_SIZE": 128}, num_warps=8),
    # 128×64 with 4 or 8 warps
    triton.Config({"Q_TILE_SIZE": 128, "K_TILE_SIZE": 64}, num_warps=4),
    triton.Config({"Q_TILE_SIZE": 128, "K_TILE_SIZE": 64}, num_warps=8),
    # maybe 128×128 with 4 or 8 warps
    triton.Config({"Q_TILE_SIZE": 128, "K_TILE_SIZE": 128}, num_warps=4),
    triton.Config({"Q_TILE_SIZE": 128, "K_TILE_SIZE": 128}, num_warps=8),
]


@triton.autotune(configs=CONFIGS_FWD, key=["D"])
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

    # Output, logsumexp, maximum for the tile - all in float32
    o_i = tl.zeros((Q_TILE_SIZE, D), dtype=tl.float32)
    l_i = tl.zeros((Q_TILE_SIZE,), dtype=tl.float32)
    m_i = tl.full((Q_TILE_SIZE,), -float("inf"), dtype=tl.float32)

    # Indices for causal masking / partial tiles
    q_indices = query_tile_index * Q_TILE_SIZE + tl.arange(0, Q_TILE_SIZE)
    q_invalid = q_indices >= N_QUERIES

    for key_tile_index in range(tl.cdiv(N_KEYS, K_TILE_SIZE)):
        # Key and value tiles
        K_tile = tl.load(K_block_ptr, boundary_check=(0, 1), padding_option="zero")
        V_tile = tl.load(V_block_ptr, boundary_check=(0, 1), padding_option="zero")

        # Logits from queries and keys
        s_ij = tl.dot(Q_tile, tl.trans(K_tile)) * scale

        # Apply the mask
        k_indices = key_tile_index * K_TILE_SIZE + tl.arange(0, K_TILE_SIZE)
        k_invalid = k_indices >= N_KEYS
        invalid = q_invalid[:, None] | k_invalid[None, :]
        if is_causal:
            invalid = invalid | (q_indices[:, None] < k_indices[None, :])
        s_ij = tl.where(invalid, -1e6, s_ij)

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
    o_i = (o_i / l_i[:, None]).to(O_block_ptr.type.element_ty)
    l_i = m_i + tl.log(l_i)

    # Write outputs and logsumexp for the query tile
    tl.store(O_block_ptr, o_i, boundary_check=(0, 1))
    tl.store(L_block_ptr, l_i, boundary_check=(0,))


@triton.autotune(
    configs=CONFIGS_BWD,
    key=["D"],
    reset_to_zero=["dQ_ptr"],  # Important!!!  We are adding and not storing!
)
@triton.jit
def flash_bck_kernel(  # type: ignore
    Q_ptr,
    K_ptr,
    V_ptr,
    dO_ptr,
    L_ptr,
    D_ptr,
    dQ_ptr,
    dK_ptr,
    dV_ptr,
    stride_qb,
    stride_qq,
    stride_qd,
    stride_kb,
    stride_kk,
    stride_kd,
    stride_vb,
    stride_vk,
    stride_vd,
    stride_dob,
    stride_doq,
    stride_dod,
    stride_lb,
    stride_lq,
    stride_db,
    stride_dq,
    stride_dqb,
    stride_dqq,
    stride_dqd,
    stride_dkb,
    stride_dkk,
    stride_dkd,
    stride_dvb,
    stride_dvk,
    stride_dvd,
    N_QUERIES,
    N_KEYS,
    scale: tl.constexpr,
    D: tl.constexpr,
    Q_TILE_SIZE: tl.constexpr,
    K_TILE_SIZE: tl.constexpr,
    is_causal: tl.constexpr,
):
    # Program indices
    key_tile_index = tl.program_id(0)
    batch_index = tl.program_id(1)

    # Block pointers
    Q_block_ptr = tl.make_block_ptr(
        Q_ptr + batch_index * stride_qb,
        shape=(N_QUERIES, D),
        strides=(stride_qq, stride_qd),
        offsets=(0, 0),
        block_shape=(Q_TILE_SIZE, D),
        order=(1, 0),
    )

    K_block_ptr = tl.make_block_ptr(
        K_ptr + batch_index * stride_kb,
        shape=(N_KEYS, D),
        strides=(stride_kk, stride_kd),
        offsets=(key_tile_index * K_TILE_SIZE, 0),
        block_shape=(K_TILE_SIZE, D),
        order=(1, 0),
    )

    V_block_ptr = tl.make_block_ptr(
        V_ptr + batch_index * stride_vb,
        shape=(N_KEYS, D),
        strides=(stride_vk, stride_vd),
        offsets=(key_tile_index * K_TILE_SIZE, 0),
        block_shape=(K_TILE_SIZE, D),
        order=(1, 0),
    )

    dO_block_ptr = tl.make_block_ptr(
        dO_ptr + batch_index * stride_dob,
        shape=(N_QUERIES, D),
        strides=(stride_doq, stride_dod),
        offsets=(0, 0),
        block_shape=(Q_TILE_SIZE, D),
        order=(1, 0),
    )

    dK_block_ptr = tl.make_block_ptr(
        dK_ptr + batch_index * stride_dkb,
        shape=(N_KEYS, D),
        strides=(stride_dkk, stride_dkd),
        offsets=(key_tile_index * K_TILE_SIZE, 0),
        block_shape=(K_TILE_SIZE, D),
        order=(1, 0),
    )

    dV_block_ptr = tl.make_block_ptr(
        dV_ptr + batch_index * stride_dvb,
        shape=(N_KEYS, D),
        strides=(stride_dvk, stride_dvd),
        offsets=(key_tile_index * K_TILE_SIZE, 0),
        block_shape=(K_TILE_SIZE, D),
        order=(1, 0),
    )

    L_block_ptr = tl.make_block_ptr(
        L_ptr + batch_index * stride_lb,
        shape=(N_QUERIES,),
        strides=(stride_lq,),
        offsets=(0,),
        block_shape=(Q_TILE_SIZE,),
        order=(0,),
    )

    D_block_ptr = tl.make_block_ptr(
        D_ptr + batch_index * stride_db,
        shape=(N_QUERIES,),
        strides=(stride_dq,),
        offsets=(0,),
        block_shape=(Q_TILE_SIZE,),
        order=(0,),
    )

    # Load K and V
    K_tile = tl.load(K_block_ptr, boundary_check=(0, 1), padding_option="zero")
    K_tile32 = K_tile.to(tl.float32)
    V_tile32 = tl.load(V_block_ptr, boundary_check=(0, 1), padding_option="zero").to(tl.float32)

    # Initialize dK and dV - all in float32 (default)
    dK_tile = tl.zeros((K_TILE_SIZE, D), dtype=tl.float32)
    dV_tile = tl.zeros((K_TILE_SIZE, D), dtype=tl.float32)

    # Indices for causal masking / partial tiles
    k_indices = key_tile_index * K_TILE_SIZE + tl.arange(0, K_TILE_SIZE)
    k_invalid = k_indices >= N_KEYS

    # Indices for updating dQ
    offs_d = tl.arange(0, D)
    dQ_tile_ptrs_base = dQ_ptr + batch_index * stride_dqb + offs_d[None, :] * stride_dqd

    for query_tile_index in range(tl.cdiv(N_QUERIES, Q_TILE_SIZE)):
        # Load Q, dO, L, D
        Q_tile = tl.load(Q_block_ptr, boundary_check=(0, 1), padding_option="zero")
        dO_tile = tl.load(dO_block_ptr, boundary_check=(0, 1), padding_option="zero")
        L_tile = tl.load(L_block_ptr, boundary_check=(0,), padding_option="zero")
        D_tile = tl.load(D_block_ptr, boundary_check=(0,), padding_option="zero")

        # Logits from queries and keys
        s_ij = tl.dot(Q_tile, tl.trans(K_tile)) * scale

        # Apply the mask
        q_indices = query_tile_index * Q_TILE_SIZE + tl.arange(0, Q_TILE_SIZE)
        q_invalid = q_indices >= N_QUERIES
        invalid = q_invalid[:, None] | k_invalid[None, :]
        if is_causal:
            invalid = invalid | (q_indices[:, None] < k_indices[None, :])
        s_ij = tl.where(invalid, -1e6, s_ij)

        # Scores
        p_ij = tl.exp(s_ij - L_tile[:, None])

        # Compute gradients
        dV_tile = tl.dot(tl.trans(p_ij), dO_tile, acc=dV_tile)
        dP_tile = tl.dot(dO_tile, tl.trans(V_tile32))
        dS_tile = p_ij * (dP_tile - D_tile[:, None]) * scale

        # Update dQ atomically using pointers
        dQ_update = tl.dot(dS_tile, K_tile32)
        dQ_tile_ptrs = dQ_tile_ptrs_base + q_indices[:, None] * stride_dqq
        tl.atomic_add(dQ_tile_ptrs, dQ_update, mask=~q_invalid[:, None], sem="relaxed")

        dK_tile = tl.dot(tl.trans(dS_tile), Q_tile.to(tl.float32), acc=dK_tile)

        # Advance block pointers
        Q_block_ptr = tl.advance(Q_block_ptr, (Q_TILE_SIZE, 0))
        dO_block_ptr = tl.advance(dO_block_ptr, (Q_TILE_SIZE, 0))
        L_block_ptr = tl.advance(L_block_ptr, (Q_TILE_SIZE,))
        D_block_ptr = tl.advance(D_block_ptr, (Q_TILE_SIZE,))

    # Write dK and dV as the j-th tiles of dK and dV
    tl.store(dK_block_ptr, dK_tile, boundary_check=(0, 1))
    tl.store(dV_block_ptr, dV_tile, boundary_check=(0, 1))


class _FlashBase(torch.autograd.Function):
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

        def grid(meta):
            return (
                triton.cdiv(n_q, meta["Q_TILE_SIZE"]),
                batch,
            )

        flash_fwd_kernel[grid](  # type: ignore
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
            is_causal=is_causal,
        )

        # Save data for the backward pass
        ctx.save_for_backward(q, k, v, o, logsumexp)  # TODO: Do I need to save in the original shapes?
        ctx.is_causal = is_causal  # type: ignore

        # Return the output
        return o


class FlashTorchBwd(_FlashBase):
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
        """Torch-compiled helper for FlashAttention 2 backward pass

        Args:
            d_o (Float[Tensor, " ... n_q d"]): Gradient of the output.
            q (Float[Tensor, " ... n_q d"]): The query.
            k (Float[Tensor, " ... n_k d"]): The key.
            v (Float[Tensor, " ... n_k d"]): The value.
            o (Float[Tensor, " ... n_q d"]): The output from the forward pass.
            logsumexp (Float[Tensor, " ... n_q"]): Logsumexp from the forward pass.
            is_causal (bool, optional): Whether to apply causal masking.  Defaults to False.

        Returns:
            Gradients of query, key, and value: dQ, dK, dV.
        """
        n_q, d_model = q.shape[-2:]
        scale = 1 / math.sqrt(d_model)

        d_o = d_o.to(torch.float32)  # Just in case
        d = einx.sum(" ... n_q d -> ... n_q", o * d_o)

        # Recompute logits from queries and keys
        s = einx.dot("... n_q d, ...  n_k d -> ... n_q n_k", q, k) * scale

        # Apply the causal mask
        if is_causal:
            mask = torch.triu(torch.ones(n_q, n_q, dtype=torch.bool, device=s.device), diagonal=1)
            s = s.masked_fill(mask, -torch.inf)

        # Recompute scores
        p = torch.exp(einx.subtract("... n_q n_k, ... n_q -> ... n_q n_k", s, logsumexp))

        # Compute gradients
        d_v = einx.dot("... n_q n_k, ... n_q d -> ... n_k d", p, d_o)
        d_p = einx.dot("... n_q d, ... n_k d -> ... n_q n_k", d_o, v)
        d_s = einx.multiply(
            "... n_q n_k, ... n_q n_k -> ... n_q n_k", p, einx.subtract("... n_q n_k, ... n_q -> ... n_q n_k", d_p, d)
        )
        d_q = einx.dot("... n_q n_k, ... n_k d -> ... n_q d", d_s, k) * scale
        d_k = einx.dot("... n_q n_k, ... n_q d -> ... n_k d", d_s, q) * scale

        # Output gradients
        return d_q, d_k, d_v

    @staticmethod
    def backward(ctx: torch.autograd.function.FunctionCtx, d_o: Float[Tensor, " ... n_q d"]):
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
        d_q, d_k, d_v = FlashTorchBwd._backward_impl(d_o, q, k, v, o, logsumexp, is_causal)
        return d_q, d_k, d_v, None


class FlashTritonBwd(_FlashBase):
    @staticmethod
    def backward(ctx: torch.autograd.function.FunctionCtx, d_o: Float[Tensor, " ... n_q d"]):
        """FlashAttention 2 backward pass in Triton

        Args:
            ctx (torch.autograd.function.FunctionCtx): Autograd context
            d_o (Float[Tensor, "... n_q d"]): Output gradient

        Returns:
            Gradients for the inputs to forward: dQ, dK, dV, None (for is_causal)
        """
        # Get saved data
        q, k, v, o, logsumexp = ctx.saved_tensors  # type: ignore
        is_causal = ctx.is_causal

        # Pre-compute d
        d_o = d_o.to(torch.float32)  # Just in case
        d = einx.sum(" ... n_q d -> ... n_q", o * d_o)

        # Flatten batch-like dimensions of inputs
        qf: Float[Tensor, " b n_q d"] = einx.rearrange("... n_q d -> (...) n_q d", q)  # type: ignore
        kf: Float[Tensor, " b n_k d"] = einx.rearrange("... n_k d -> (...) n_k d", k)  # type: ignore
        vf: Float[Tensor, " b n_k d"] = einx.rearrange("... n_k d -> (...) n_k d", v)  # type: ignore
        d_of: Float[Tensor, " b n_q d"] = einx.rearrange("... n_q d -> (...) n_q d", d_o)  # type: ignore
        lf: Float[Tensor, " b n_q"] = einx.rearrange("... n_q -> (...) n_q", logsumexp)  # type: ignore
        df: Float[Tensor, " b n_q"] = einx.rearrange("... n_q -> (...) n_q", d)  # type: ignore

        # Input dimensions
        batch, n_q, d_model = qf.shape
        n_k = kf.shape[-2]

        assert q.is_cuda and k.is_cuda and v.is_cuda and o.is_cuda and logsumexp.is_cuda and d_o.is_cuda, (
            "Expected CUDA tensors"
        )
        assert (
            q.is_contiguous()
            and k.is_contiguous()
            and v.is_contiguous()
            and o.is_contiguous()
            and logsumexp.is_contiguous()
            and d_o.is_contiguous()
        ), "Expect contiguous tensors"
        if is_causal:
            assert n_q == n_k, "Expect equal sequence (query, key, value) lengths when is_causal is True"

        # Prepare the outputs
        d_q = torch.zeros_like(q, dtype=torch.float32)
        d_k = torch.empty_like(k, dtype=torch.float32)
        d_v = torch.empty_like(v, dtype=torch.float32)
        d_qf: Float[Tensor, " b n_q d"] = einx.rearrange("... n_q d -> (...) n_q d", d_q, b=batch, n_q=n_q, d=d_model)  # type: ignore
        d_kf: Float[Tensor, " b n_k d"] = einx.rearrange("... n_k d -> (...) n_k d", d_k, b=batch, n_k=n_k, d=d_model)  # type: ignore
        d_vf: Float[Tensor, " b n_k d"] = einx.rearrange("... n_k d -> (...) n_k d", d_v, b=batch, n_k=n_k, d=d_model)  # type: ignore

        scale = 1.0 / math.sqrt(d_model)

        def grid(meta):
            return (
                triton.cdiv(n_k, meta["K_TILE_SIZE"]),
                batch,
            )

        flash_bck_kernel[grid](  # type: ignore
            qf,
            kf,
            vf,
            d_of,
            lf,
            df,
            d_qf,
            d_kf,
            d_vf,
            *qf.stride(),
            *kf.stride(),
            *vf.stride(),
            *d_of.stride(),
            *lf.stride(),
            *df.stride(),
            *d_qf.stride(),
            *d_kf.stride(),
            *d_vf.stride(),
            N_QUERIES=n_q,
            N_KEYS=n_k,
            scale=scale,
            D=d_model,
            is_causal=is_causal,
        )

        # Output gradients
        return d_q, d_k, d_v, None
