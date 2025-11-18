import math

import einx
import torch
from jaxtyping import Float
from torch import Tensor

B_q = 16
B_k = 16


class FlashPytorch(torch.autograd.Function):
    @staticmethod
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
    def backward(ctx: torch.autograd.function.FunctionCtx, grad_out: Float[Tensor, " ... n_q d"]):
        raise NotImplementedError
