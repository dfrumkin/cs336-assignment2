import math

import einx
import torch
from beartype import beartype
from jaxtyping import Float, jaxtyped
from torch import Tensor

B_q = 16
B_k = 16


class FlashPytorch(torch.autograd.Function):
    @staticmethod
    @jaxtyped(typechecker=beartype)
    def forward(
        ctx: torch.autograd.function.FunctionCtx,
        q: Float[Tensor, " ... n_q d"],
        k: Float[Tensor, " ... n_k d"],
        v: Float[Tensor, " ... n_k d"],
        is_causal: bool = False,
    ) -> Float[Tensor, " ... n_q d"]:
        """FlashAttention 2 forward pass in pytorch

        Args:
            ctx (torch.autograd.function.FunctionCtx): Autograd context
            q (Float[Tensor, " ... n_q d"]): Query
            k (Float[Tensor, " ... n_k d"]): Key
            v (Float[Tensor, " ... n_k d"]): Value
            is_causal (bool, optional): Whether to apply causal masking.  Defaults to False.
        Returns:
            Float[Tensor, " ... n_q d"]: Output tensor
        """
        device = q.device
        d_model = q.shape[-1]
        scale = 1 / math.sqrt(d_model)

        # Output and logsumexp
        o = torch.empty(q.shape, device=device)
        logsumexp = torch.empty(q.shape[:-1], device=device)

        for i in range(0, q.shape[-2], B_q):
            # Query tile
            q_i = q[..., i : i + B_q, :]

            # Output, logsumexp, maximum for the tile
            o_i = torch.zeros((*q.shape[:-2], B_q, d_model), device=device)
            l_i = torch.zeros((*q.shape[:-2], B_q), device=device)
            m_i = torch.full((*q.shape[:-2], B_q), -torch.inf, device=device)

            for j in range(0, k.shape[-2], B_k):
                # Key and value tiles
                k_j = k[..., j : j + B_k, :]
                v_j = v[..., j : j + B_k, :]

                # Logits from queries and keys
                s_ij = einx.dot("... B_q d, ...  B_k d -> ... B_q B_k", q_i, k_j, B_q=B_q, B_k=B_k, d=d_model) * scale

                # The new maximum and the associated correction factor
                m_ij = torch.maximum(m_i, s_ij.max(dim=-1).values)
                correction_factor = torch.exp(m_i - m_ij)
                m_i = m_ij

                # Scores
                p_ij = torch.exp(einx.subtract("... B_q B_k, ... B_q -> ... B_q B_k", s_ij, m_i))

                # Denominator
                l_i = correction_factor * l_i + einx.sum("... B_q B_k -> ... B_q", p_ij, B_q=B_q, B_k=B_k)

                # Numerator
                o_i = einx.multiply(
                    "... B_q, ... B_q d -> ... B_q d", correction_factor, o_i, B_q=B_q, d=d_model
                ) + einx.dot("... B_q B_k, ... B_k d -> ... B_q d", p_ij, v_j, B_q=B_q, B_k=B_k, d=d_model)

            # Compute the output and logsumexp for the i'th query tile
            o_i = einx.divide("... B_q d, ... B_q -> ... B_q d", o_i, l_i)
            l_i = m_i + torch.log(l_i)

            # Write outputs and logsumexp for the query tile
            o[..., i : i + B_q, :] = o_i
            logsumexp[..., i : i + B_q] = l_i

        # Save the inputs, the output, and the logsumexp for the backward pass
        ctx.save_for_backward(q, k, v, o, logsumexp)

        # Return the output
        return o

    @staticmethod
    @jaxtyped(typechecker=beartype)
    def backward(ctx: torch.autograd.function.FunctionCtx, d_o: Float[Tensor, " ... n_q d"]):
        """FlashAttention 2 backward pass in pytorch

        Args:
            ctx (torch.autograd.function.FunctionCtx): Autograd context
            d_o (Float[Tensor, "... n_q d"]): Output gradient

        Returns:
            Gradients for the inputs to forward: dQ, dK, dV, None (for is_causal)
        """
        q, k, v, o, logsumexp = ctx.saved_tensors  # type: ignore
        n_q, d_model = q.shape[-2:]
        n_k = k.shape[-2]
        scale = 1 / math.sqrt(d_model)

        d_o = d_o.to(torch.float32)  # Just in case
        d = einx.sum(" ... n_q d -> ... n_q", o * d_o)

        # Recompute logits from queries and keys
        s = einx.dot("... n_q d, ...  n_k d -> ... n_q n_k", q, k, n_q=n_q, n_k=n_k, d=d_model) * scale

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

        # Output gradients for the inputs of forward: q, k, v, is_causal
        return d_q, d_k, d_v, None
