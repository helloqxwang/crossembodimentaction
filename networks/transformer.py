from typing import Optional, Sequence
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import numpy as np


def _build_sinusoidal_positional_encoding(
    max_length: int,
    dim: int,
    device: torch.device | None = None,
    dtype: torch.dtype | None = None,
) -> torch.Tensor:
    """Create a standard transformer sinusoidal positional encoding.

    This mirrors the formulation from *Attention Is All You Need* and avoids
    any learnable parameters, keeping the module lightweight and robust for
    small data regimes.
    """

    if dtype is None:
        dtype = torch.float32

    position = torch.arange(max_length, device=device, dtype=dtype).unsqueeze(1)
    div_term = torch.exp(
        torch.arange(0, dim, 2, device=device, dtype=dtype)
        * (-torch.log(torch.tensor(10000.0, device=device, dtype=dtype)) / dim)
    )
    pe = torch.zeros(max_length, dim, device=device, dtype=dtype)
    pe[:, 0::2] = torch.sin(position * div_term)
    pe[:, 1::2] = torch.cos(position * div_term)
    return pe


class SinusoidalPositionalEncoding(nn.Module):
    """Sinusoidal positional encoding with optional dropout."""

    def __init__(self, dim: int, max_length: int = 512, dropout: float = 0.0) -> None:
        super().__init__()
        self.dim = dim
        self.max_length = max_length
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        pe = _build_sinusoidal_positional_encoding(max_length, dim)
        self.register_buffer("pe", pe, persistent=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Add positional encoding to ``x``.

        Parameters
        ----------
        x: ``Tensor`` of shape ``(batch, length, dim)``.
        """

        if x.size(1) > self.max_length:
            raise ValueError(
                f"Sequence length {x.size(1)} exceeds configured max_length="
                f"{self.max_length}. Increase max_length when constructing the encoder."
            )

        pe = self.pe[: x.size(1)].unsqueeze(0)
        x = x + pe
        return self.dropout(x)


class TransformerEncoder(nn.Module):
    """A configurable stack of transformer encoder layers for token inputs."""

    def __init__(
        self,
        *,
        input_dim: int,
        model_dim: int = 128,
        output_dim: Optional[int] = None,
        num_heads: int = 4,
        num_layers: int = 2,
        mlp_ratio: float = 4.0,
        dropout: float = 0.1,
        attn_dropout: float = 0.0,
        activation: str = "gelu",
        use_positional_encoding: bool = True,
        max_length: int = 512,
        norm_first: bool = False,
        final_layer_norm: bool = True,
    ) -> None:
        super().__init__()

        if input_dim <= 0:
            raise ValueError("input_dim must be positive")
        if model_dim % num_heads != 0:
            raise ValueError("model_dim must be divisible by num_heads")
        if max_length <= 0:
            raise ValueError("max_length must be positive")

        if output_dim is None:
            output_dim = model_dim

        self.model_dim = model_dim
        self.max_length = max_length
        self.use_positional_encoding = use_positional_encoding

        self.input_proj = nn.Identity() if input_dim == model_dim else nn.Linear(input_dim, model_dim)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=model_dim,
            nhead=num_heads,
            dim_feedforward=int(model_dim * mlp_ratio),
            dropout=dropout,
            activation=activation,
            batch_first=True,
            norm_first=norm_first,
        )
        if attn_dropout > 0:
            encoder_layer.self_attn.dropout = attn_dropout

        self.encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_layers,
            norm=nn.LayerNorm(model_dim) if final_layer_norm else None,
        )

        self.positional_encoding: Optional[nn.Module]
        if use_positional_encoding:
            self.positional_encoding = SinusoidalPositionalEncoding(
                dim=model_dim,
                max_length=max_length,
                dropout=dropout,
            )
        else:
            self.positional_encoding = None

        self.output_proj = nn.Identity() if output_dim == model_dim else nn.Linear(model_dim, output_dim)

        self._cached_causal_mask: torch.Tensor | None = None

    def forward(
        self,
        x: torch.Tensor,
        *,
        key_padding_mask: Optional[torch.Tensor] = None,
        attn_mask: Optional[torch.Tensor] = None,
        causal: bool = False,
    ) -> torch.Tensor:
        """Encode a batch of token sequences."""

        if x.dim() != 3:
            raise ValueError(f"Expected (batch, length, dim) input, got shape {tuple(x.shape)}")

        _, seq_len, _ = x.shape

        x = self.input_proj(x)

        if self.positional_encoding is not None:
            x = self.positional_encoding(x)

        causal_mask = self._maybe_build_causal_mask(seq_len, x.device, x.dtype) if causal else None
        if causal_mask is not None:
            attn_mask = causal_mask if attn_mask is None else attn_mask + causal_mask

        x = self.encoder(
            x,
            mask=attn_mask,
            src_key_padding_mask=key_padding_mask,
            is_causal=False,
        )

        return self.output_proj(x)

    def _maybe_build_causal_mask(
        self, seq_len: int, device: torch.device, dtype: torch.dtype
    ) -> torch.Tensor:
        """Return a cached causal (upper-triangular) mask for the given length."""

        if self._cached_causal_mask is not None:
            cached_len = self._cached_causal_mask.size(-1)
            if cached_len >= seq_len and self._cached_causal_mask.device == device:
                return self._cached_causal_mask[:seq_len, :seq_len]

        mask_dtype = dtype if torch.is_floating_point(torch.empty((), device=device, dtype=dtype)) else torch.float32
        mask = torch.full((seq_len, seq_len), float("-inf"), device=device, dtype=mask_dtype)
        mask = torch.triu(mask, diagonal=1)
        self._cached_causal_mask = mask
        return mask


# ---------------------------------------------------------------------------
# Attention attribution utilities
# ---------------------------------------------------------------------------

def compute_token_dependency(
    model: TransformerEncoder,
    x: torch.Tensor,
    *,
    key_padding_mask: Optional[torch.Tensor] = None,
    attn_mask: Optional[torch.Tensor] = None,
    causal: bool = False,
) -> torch.Tensor:
    """Compute aggregated token-to-token dependency via attention rollout.

    Returns a ``(batch, seq_len, seq_len)`` tensor where entry ``(b, i, j)``
    approximates how much input token ``j`` contributes to output token ``i``
    after compounding attention across all encoder layers. This follows the
    attention rollout scheme (Abnar & Zuidema, 2020), using the averaged
    attention weights available from PyTorch's ``MultiheadAttention``.
    """

    if x.dim() != 3:
        raise ValueError(f"Expected (batch, length, dim) input, got {tuple(x.shape)}")

    attn_maps: list[torch.Tensor] = []

    def _save_attn(_: nn.Module, __: tuple, output: tuple) -> None:
        # MultiheadAttention returns (attn_output, attn_weights) when need_weights=True
        if isinstance(output, tuple) and len(output) == 2 and output[1] is not None:
            attn_maps.append(output[1].detach())

    # Temporarily force need_weights=True so hooks receive attn maps
    patched_forwards = []
    handles = []
    try:
        for layer in model.encoder.layers:
            mha = layer.self_attn
            orig_forward = mha.forward

            def _patched_forward(*args, orig_forward=orig_forward, **kwargs):
                kwargs.setdefault("need_weights", True)
                kwargs.setdefault("average_attn_weights", True)
                return orig_forward(*args, **kwargs)

            patched_forwards.append((mha, orig_forward))
            mha.forward = _patched_forward  # type: ignore[assignment]
            handles.append(mha.register_forward_hook(_save_attn))

        # Force math attention (disable flash/mem-efficient) so PyTorch returns weights.
        sdp_ctx = (
            torch.backends.cuda.sdp_kernel(enable_flash=False, enable_mem_efficient=False, enable_math=True)
            if torch.cuda.is_available()
            else torch.backends.cuda.sdp_kernel()
        )
        with sdp_ctx, torch.no_grad():
            _ = model(
                x,
                key_padding_mask=key_padding_mask,
                attn_mask=attn_mask,
                causal=causal,
            )
    finally:
        for h in handles:
            h.remove()
        for mha, orig_forward in patched_forwards:
            mha.forward = orig_forward  # restore

    if not attn_maps:
        raise RuntimeError("No attention maps were captured; verify model has encoder layers.")

    # attn_maps: list of (batch, seq_len, seq_len) averaged over heads
    seq_len = attn_maps[0].size(-1)
    batch = attn_maps[0].size(0)
    device = attn_maps[0].device

    rollout = torch.eye(seq_len, device=device).unsqueeze(0).expand(batch, -1, -1)

    for attn in attn_maps:
        # Add residual connection per rollout definition
        attn = attn + torch.eye(seq_len, device=device)
        attn = attn / attn.sum(dim=-1, keepdim=True).clamp_min(1e-6)
        rollout = torch.bmm(attn, rollout)

    # Optional masking: zero-out padded tokens' influence on outputs
    if key_padding_mask is not None:
        mask = (~key_padding_mask).float().unsqueeze(1)  # (B,1,L)
        rollout = rollout * mask

    return rollout


def plot_token_dependency(
    dependency: torch.Tensor | np.ndarray,
    *,
    token_labels: Optional[Sequence[str]] = None,
    title: str = "Token dependency",
    figsize: tuple[float, float] = (5.0, 4.0),
    cmap: str = "viridis",
) -> tuple[plt.Figure, plt.Axes]:
    """Plot a heatmap for a ``(seq_len, seq_len)`` dependency matrix."""

    if isinstance(dependency, torch.Tensor):
        dep = dependency.detach().cpu()
        if dep.dim() == 3:
            dep = dep[0]  # take first sample if batch provided
        dependency_np = dep.numpy()
    else:
        dependency_np = np.asarray(dependency)

    if dependency_np.ndim != 2 or dependency_np.shape[0] != dependency_np.shape[1]:
        raise ValueError("dependency must be square (seq_len x seq_len)")

    seq_len = dependency_np.shape[0]
    if token_labels is None:
        token_labels = [f"t{i}" for i in range(seq_len)]

    fig, ax = plt.subplots(figsize=figsize)
    im = ax.imshow(dependency_np, cmap=cmap, origin="lower")
    ax.set_xticks(range(seq_len))
    ax.set_yticks(range(seq_len))
    ax.set_xticklabels(token_labels, rotation=45, ha="right")
    ax.set_yticklabels(token_labels)
    ax.set_xlabel("Input tokens")
    ax.set_ylabel("Output tokens")
    ax.set_title(title)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="Contribution")
    fig.tight_layout()
    return fig, ax


if __name__ == "__main__":
    torch.manual_seed(0)
    encoder = TransformerEncoder(input_dim=64, model_dim=128, num_layers=2, num_heads=4)
    x = torch.randn(2, 6, 64)

    # Mask out last two tokens of the second sequence
    key_padding_mask = torch.tensor([[False, False, False, False, False, False], [False, False, False, True, True, True]])

    out = encoder(x, key_padding_mask=key_padding_mask, causal=True)
    print("Output shape:", out.shape)

    # Causal run for autoregressive setups
    causal_out = encoder(x, causal=True)
    print("Causal output shape:", causal_out.shape)