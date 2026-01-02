import math
from dataclasses import dataclass

import torch


@dataclass
class ModelConfig:
    num_hidden_layers: int = 36
    num_experts: int = 128
    experts_per_token: int = 4
    vocab_size: int = 201088
    hidden_size: int = 2880
    intermediate_size: int = 2880
    swiglu_limit: float = 7.0
    head_dim: int = 64
    num_attention_heads: int = 64
    num_key_value_heads: int = 8
    sliding_window: int = 128
    initial_context_length: int = 4096
    rope_theta: float = 150000.0
    rope_scaling_factor: float = 32.0
    rope_ntk_alpha: float = 1.0
    rope_ntk_beta: float = 32.0
    num_levels: int = 3
    num_nbrs: int = 64
    nbrs_per_token: int = 16

class RMSNorm(torch.nn.Module):
    def __init__(
        self, num_features: int, eps: float = 1e-05, device: torch.device | None = None
    ):
        super().__init__()
        self.num_features = num_features
        self.eps = eps
        self.scale = torch.nn.Parameter(
            torch.ones(num_features, device=device, dtype=torch.float32)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        assert x.shape[-1] == self.num_features
        t, dtype = x.float(), x.dtype
        t = t * torch.rsqrt(torch.mean(t**2, dim=-1, keepdim=True) + self.eps)
        return (t * self.scale).to(dtype)


def _apply_rotary_emb(
    x: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
) -> torch.Tensor:
    cos = cos.unsqueeze(-2).to(x.dtype)
    sin = sin.unsqueeze(-2).to(x.dtype)
    x1, x2 = torch.chunk(x, 2, dim=-1)
    o1 = x1 * cos - x2 * sin
    o2 = x2 * cos + x1 * sin
    return torch.cat((o1, o2), dim=-1)


class RotaryEmbedding(torch.nn.Module):
    def __init__(
        self,
        head_dim: int,
        base: int,
        dtype: torch.dtype,
        initial_context_length: int = 4096,
        scaling_factor: float = 1.0,
        ntk_alpha: float = 1.0,
        ntk_beta: float = 32.0,
        device: torch.device | None = None,
    ) -> None:
        super().__init__()
        self.head_dim = head_dim
        self.base = base
        self.dtype = dtype
        self.initial_context_length = initial_context_length
        self.scaling_factor = scaling_factor
        self.ntk_alpha = ntk_alpha
        self.ntk_beta = ntk_beta
        self.device = device

    def _compute_concentration_and_inv_freq(self) -> torch.Tensor:
        """See YaRN paper: https://arxiv.org/abs/2309.00071"""
        freq = self.base ** (
            torch.arange(0, self.head_dim, 2, dtype=torch.float, device=self.device)
            / self.head_dim
        )
        if self.scaling_factor > 1.0:
            concentration = (
                0.1 * math.log(self.scaling_factor) + 1.0
            )  # YaRN concentration

            d_half = self.head_dim / 2
            # NTK by parts
            low = (
                d_half
                * math.log(self.initial_context_length / (self.ntk_beta * 2 * math.pi))
                / math.log(self.base)
            )
            high = (
                d_half
                * math.log(self.initial_context_length / (self.ntk_alpha * 2 * math.pi))
                / math.log(self.base)
            )
            assert 0 < low < high < d_half - 1

            interpolation = 1.0 / (self.scaling_factor * freq)
            extrapolation = 1.0 / freq

            ramp = (
                torch.arange(d_half, dtype=torch.float32, device=freq.device) - low
            ) / (high - low)
            mask = 1 - ramp.clamp(0, 1)

            inv_freq = interpolation * (1 - mask) + extrapolation * mask
        else:
            concentration = 1.0
            inv_freq = 1.0 / freq

        return concentration, inv_freq

    def _compute_cos_sin(self, num_tokens: int):
        concentration, inv_freq = self._compute_concentration_and_inv_freq()
        t = torch.arange(num_tokens, dtype=torch.float32, device=self.device)
        freqs = torch.einsum("i,j->ij", t, inv_freq)
        cos = freqs.cos() * concentration
        sin = freqs.sin() * concentration
        return cos, sin

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        num_tokens = query.shape[0]
        cos, sin = self._compute_cos_sin(num_tokens)

        query_shape = query.shape
        query = query.view(num_tokens, -1, self.head_dim)
        query = _apply_rotary_emb(query, cos, sin)
        query = query.reshape(query_shape)

        key_shape = key.shape
        key = key.view(num_tokens, -1, self.head_dim)
        key = _apply_rotary_emb(key, cos, sin)
        key = key.reshape(key_shape)
        return query, key


def sdpa(Q, K, V, S, sm_scale, sliding_window=0):
    # sliding_window == 0 means no sliding window
    n_tokens, n_heads, q_mult, d_head = Q.shape
    assert K.shape == (n_tokens, n_heads, d_head)
    assert V.shape == (n_tokens, n_heads, d_head)
    K = K[:, :, None, :].expand(-1, -1, q_mult, -1)
    V = V[:, :, None, :].expand(-1, -1, q_mult, -1)
    S = S.reshape(n_heads, q_mult, 1, 1).expand(-1, -1, n_tokens, -1)
    mask = torch.triu(Q.new_full((n_tokens, n_tokens), -float("inf")), diagonal=1)
    if sliding_window > 0:
        mask += torch.tril(
            mask.new_full((n_tokens, n_tokens), -float("inf")), diagonal=-sliding_window
        )
    QK = torch.einsum("qhmd,khmd->hmqk", Q, K)
    QK *= sm_scale
    QK += mask[None, None, :, :]
    QK = torch.cat([QK, S], dim=-1)
    W = torch.softmax(QK, dim=-1)
    W = W[..., :-1]
    attn = torch.einsum("hmqk,khmd->qhmd", W, V)
    return attn.reshape(n_tokens, -1)


class AttentionBlock(torch.nn.Module):
    def __init__(
        self,
        config: ModelConfig,
        layer_idx: int = 0,
        device: torch.device | None = None,
    ):
        super().__init__()
        self.head_dim = config.head_dim
        self.num_attention_heads = config.num_attention_heads
        self.num_key_value_heads = config.num_key_value_heads
        # Only apply sliding window to every other layer
        self.sliding_window = config.sliding_window if layer_idx % 2 == 0 else 0
        self.sinks = torch.nn.Parameter(
            torch.empty(config.num_attention_heads, device=device, dtype=torch.bfloat16)
        )
        self.norm = RMSNorm(config.hidden_size, device=device)
        qkv_dim = config.head_dim * (
            config.num_attention_heads + 2 * config.num_key_value_heads
        )
        self.qkv = torch.nn.Linear(
            config.hidden_size, qkv_dim, device=device, dtype=torch.bfloat16
        )
        self.out = torch.nn.Linear(
            config.head_dim * config.num_attention_heads,
            config.hidden_size,
            device=device,
            dtype=torch.bfloat16,
        )
        self.sm_scale = 1 / math.sqrt(config.head_dim)
        self.rope = RotaryEmbedding(
            config.head_dim,
            config.rope_theta,
            torch.float32,
            initial_context_length=config.initial_context_length,
            scaling_factor=config.rope_scaling_factor,
            ntk_alpha=config.rope_ntk_alpha,
            ntk_beta=config.rope_ntk_beta,
            device=device,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        t = self.norm(x)
        qkv = self.qkv(t)
        q = qkv[:, : self.num_attention_heads * self.head_dim].contiguous()
        k = qkv[
            :,
            self.num_attention_heads
            * self.head_dim : (self.num_attention_heads + self.num_key_value_heads)
            * self.head_dim,
        ].contiguous()
        v = qkv[
            :,
            (self.num_attention_heads + self.num_key_value_heads)
            * self.head_dim : (self.num_attention_heads + 2 * self.num_key_value_heads)
            * self.head_dim,
        ].contiguous()

        q = q.view(
            -1,
            self.num_key_value_heads,
            self.num_attention_heads // self.num_key_value_heads,
            self.head_dim,
        )
        k = k.view(-1, self.num_key_value_heads, self.head_dim)
        v = v.view(-1, self.num_key_value_heads, self.head_dim)
        q, k = self.rope(q, k)
        t = sdpa(q, k, v, self.sinks, self.sm_scale, self.sliding_window)
        t = self.out(t)
        t = x + t
        return t


def swiglu(x, alpha: float = 1.702, limit: float = 7.0):
    x_glu, x_linear = x[..., ::2], x[..., 1::2]
    # Clamp the input values
    x_glu = x_glu.clamp(min=None, max=limit)
    x_linear = x_linear.clamp(min=-limit, max=limit)
    out_glu = x_glu * torch.sigmoid(alpha * x_glu)
    # Note we add an extra bias of 1 to the linear layer
    return out_glu * (x_linear + 1)


class HNSWBlock(torch.nn.Module):
    """
    An MLP block where active weights are retrieved by an HNSW-inspired hierarchical
    search.

    The MLP weights are viewed as a large set of (key, value) vector pairs. The input x
    is seen as a batch of queries. The goal is to retrieve the sparse set of active
    (key, value) weights for each query.

    The keys are organized as a hierarchical tree over the unit sphere. Starting from
    the root level, we find the top k closest root keys for each query. We then descend
    the tree, continuing to take the top k keys at each level. At the bottom level, the
    remaining keys and corresponding values are selected as active.

    This scheme can be seen as a generalization of the classic MoE MLP block,
    corresponding the special case of only two levels.
    """
    def __init__(
        self,
        config: ModelConfig,
        device: torch.device | None = None,
    ):
        super().__init__()
        # number of levels in the tree
        # classic moe: 2 levels
        self.num_levels = config.num_levels
        # number of neighbors branching at each level. also the number of root nodes,
        # for convenience. we could consider generalizing to arbitrary number of nodes
        # at each level, but this is ok for now.
        # classic moe: num_experts root nodes, and hidden_size number of neighbors
        self.num_nbrs = config.num_nbrs
        # number of neighbor nodes to retrieve at each level
        # classic moe: nbrs_per_token = experts_per_token
        self.nbrs_per_token = config.nbrs_per_token
        self.norm = RMSNorm(config.hidden_size, device=device)
        # gelu activation for convenience for now
        self.gelu = torch.nn.GELU()
        # gating keys for the top-down search of the tree.
        # you have num_nbrs ^ i nodes at level i
        self.gate_weights = torch.nn.ParameterList(
            torch.nn.Parameter(
                torch.empty(
                    (self.num_nbrs,) * (ii + 1) + (config.hidden_size,),
                    device=device,
                    dtype=torch.bfloat16,
                )
            )
            for ii in range(self.num_levels - 1)
        )
        # the input mlp weights, ie key weights, are at the bottom level.
        self.mlp1_weight = torch.nn.Parameter(
            torch.empty(
                (self.num_nbrs,) * self.num_levels + (config.hidden_size,),
                device=device,
                dtype=torch.bfloat16,
            )
        )
        # the output mlp weights, ie value weights. same number as keys.
        self.mlp2_weight = torch.nn.Parameter(
            torch.empty(
                (self.num_nbrs,) * self.num_levels + (config.hidden_size,),
                device=device,
                dtype=torch.bfloat16,
            )
        )
        # a scale parameter, not sure if we need it. but will see why later.
        self.scale = torch.nn.Parameter(torch.empty((), device=device, dtype=torch.bfloat16))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C = x.shape
        t = self.norm(x)

        gates = list(self.gate_weights)
        k = self.mlp1_weight
        v = self.mlp2_weight

        # search the tree
        # at each level, the actual node vectors are constructed by adding the gate
        # weights as a small residual on top of the parent nodes. this is what creates
        # the tree structure.
        parent = None
        for ii in range(self.num_levels - 1):
            gate = gates[ii]
            # add dummy batch dimension
            if ii == 0:
                gate = expand_dim(gate, 0, B)
            # construct the node as residual of parent + scale * weight
            gate = residual_gate(gate, parent=parent, level=ii)
            # select the top k nearest nodes at this level
            indices = apply_gate(gate, t, k=self.nbrs_per_token)
            # prune the tree for the lower levels still to search
            for jj in range(ii + 1, self.num_levels - 1):
                gates[jj] = gather_values(indices, gates[jj])
            # prune the lowest level keys and values
            k = gather_values(indices, k)
            v = gather_values(indices, v)
            # update the parent nodes for the next level
            parent = gate

        # final keys as a residual for the lowest level of the tree
        k = residual_gate(k, parent=parent, level=self.num_levels-1)

        # after selection, now just apply mlp as usual
        k = k.reshape(B, -1, C)
        v = v.reshape(B, -1, C)
        t = torch.einsum("bc,bdc->bd", t, k)
        # this is where that scale parameter comes in
        # the issue is that the keys are unit norm, which might not be best going into
        # the gelu.
        # todo: think about this scaling
        t = self.gelu(self.scale * t)
        t = torch.einsum("bd,bdc->bc", t, v)

        return x + t


def apply_gate(gate: torch.Tensor, x: torch.Tensor, k: int):
    # select the top k nodes
    # gate: (b, k, k, ..., v, c)
    # x: (b, c)
    # returns:
    # indices: (b, k, k, ..., k)
    B, *shape, C = gate.shape
    B, C = x.shape
    gate = gate.reshape(B, -1, C)
    g = torch.einsum("bc,bvc->bv", x, gate)
    g = g.reshape((B, *shape))
    _, indices = torch.topk(g, k=k, dim=-1, sorted=True)
    return indices


def residual_gate(gate: torch.Tensor, parent: torch.Tensor | None = None, level: int = 0):
    # update gate as gate + scale * parent
    # effectively we are trying to construct a dense multi-scale mesh over the sphere of
    # progressively finer scales.
    # inputs:
    # gate: (*, v, c)
    # parent: (*, c)
    if parent is not None:
        dim = gate.shape[-1]
        # ensure parent is on sphere
        parent = torch.nn.functional.normalize(parent, dim=-1)
        # scale of small residual to ensure a target angular separation
        # todo: think more about this scaling
        theta = math.pi / 3 / (1.5 ** level)
        scale = math.sqrt((1 / dim) * (1 / math.cos(theta) ** 2 - 1))
        gate = parent.unsqueeze(-2) + scale * gate
    gate = torch.nn.functional.normalize(gate, dim=-1)
    return gate


def gather_values(indices: torch.Tensor, values: torch.Tensor):
    # gather values over the last index dimension
    # values can have extra trailing dimensions which we expand to
    # indices: (b, k, ..., k)
    # values: (k, ..., v, ...)
    B, *shape = indices.shape
    extra_shape = values.shape[len(shape):]
    indices = indices.reshape(indices.shape + (1,) * len(extra_shape))
    indices = indices.expand(indices.shape + extra_shape)
    values = expand_dim(values, 0, B)
    values = torch.gather(values, dim=indices.ndim-1, index=indices)
    return values


def expand_dim(x: torch.Tensor, dim: int, size: int):
    shape = list(x.shape)
    shape.insert(dim=size)
    return x.unsqueeze(dim).expand(shape)


class TransformerBlock(torch.nn.Module):
    def __init__(
        self,
        config: ModelConfig,
        layer_idx: int,
        device: torch.device | None = None,
    ):
        super().__init__()
        self.layer_idx = layer_idx
        self.attn = AttentionBlock(config, layer_idx, device)
        self.mlp = HNSWBlock(config, device)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.attn(x)
        x = self.mlp(x)
        return x


class Transformer(torch.nn.Module):
    def __init__(
        self,
        config: ModelConfig,
        device: torch.device | None = None,
    ):
        super().__init__()
        self.embedding = torch.nn.Embedding(
            config.vocab_size, config.hidden_size, device=device, dtype=torch.bfloat16
        )
        self.block = torch.nn.ModuleList(
            [
                TransformerBlock(config, layer_idx, device)
                for layer_idx in range(config.num_hidden_layers)
            ]
        )
        self.norm = RMSNorm(config.hidden_size, device=device)
        self.unembedding = torch.nn.Linear(
            config.hidden_size,
            config.vocab_size,
            bias=False,
            device=device,
            dtype=torch.bfloat16,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.embedding(x)
        for block in self.block:
            x = block(x)
        x = self.norm(x)
        x = self.unembedding(x)
        return x


class TokenGenerator:
    @torch.inference_mode()
    def __init__(self, model: Transformer, device: torch.device):
        self.model = model.to(device)
        self.device = device

    @torch.inference_mode()
    def generate(self,
                 prompt_tokens: list[int],
                 stop_tokens: list[int],
                 temperature: float = 1.0,
                 max_tokens: int = 0,
                 return_logprobs: bool = False):
        tokens = list(prompt_tokens)
        num_generated_tokens = 0
        while max_tokens == 0 or num_generated_tokens < max_tokens:
            logits = self.model(torch.as_tensor(tokens, dtype=torch.int32, device=self.device))[-1]
            if temperature == 0.0:
                predicted_token = torch.argmax(logits, dim=-1).item()
            else:
                probs = torch.softmax(logits * (1.0 / temperature), dim=-1)
                predicted_token = torch.multinomial(probs, num_samples=1).item()
            tokens.append(predicted_token)
            num_generated_tokens += 1

            if return_logprobs:
                logprobs = torch.log_softmax(logits, dim=-1)
                selected_logprobs = logprobs[predicted_token].item()
                yield predicted_token, selected_logprobs
            else:
                yield predicted_token

            if predicted_token in stop_tokens:
                break
