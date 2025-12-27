import torch
import torch.nn as nn
from torch.nn.utils import spectral_norm


def _maybe_sn(module: nn.Module, enabled: bool) -> nn.Module:
    return spectral_norm(module) if enabled else module


def _norm_layer(kind: str, num_features: int) -> nn.Module:
    kind = kind.lower()
    if kind in ("none", "identity"):
        return nn.Identity()
    if kind in ("batch", "batchnorm", "bn"):
        return nn.BatchNorm2d(num_features)
    if kind in ("instance", "instancenorm", "in"):
        return nn.InstanceNorm2d(num_features, affine=True)
    raise ValueError(f"Unknown norm kind: {kind!r}")


class Discriminator(nn.Module):
    """Hybrid CNN + MLP discriminator.

    - Input: (N, 1, 256, 256)
    - Output: (N, 1, 1, 1) (kept for backward compatibility)

    Uses a CNN feature extractor, then combines:
    - A convolutional head (PatchGAN-style, but reduced to a single logit)
    - An MLP head over pooled features
    """

    def __init__(
        self,
        in_channels: int = 1,
        base_channels: int = 64,
        n_downsamples: int = 6,
        max_channels: int = 512,
        use_spectral_norm: bool = True,
        norm: str | None = None,
        negative_slope: float = 0.2,
        mlp_hidden: tuple[int, ...] = (512, 256),
        mlp_dropout: float = 0.0,
        use_conv_head: bool = True,
        use_mlp_head: bool = True,
        pool_to_4x4: bool = True,
    ):
        super().__init__()

        if in_channels <= 0:
            raise ValueError("in_channels must be > 0")
        if base_channels <= 0:
            raise ValueError("base_channels must be > 0")
        if n_downsamples < 1:
            raise ValueError("n_downsamples must be >= 1")
        if max_channels <= 0:
            raise ValueError("max_channels must be > 0")
        if not (use_conv_head or use_mlp_head):
            raise ValueError("At least one of use_conv_head/use_mlp_head must be True")
        if mlp_dropout < 0:
            raise ValueError("mlp_dropout must be >= 0")

        # With spectral norm, BatchNorm can sometimes destabilize training.
        # Default to no normalization unless the caller explicitly asks for it.
        norm_kind = (norm or "none")

        layers: list[nn.Module] = []

        # First block: no norm.
        out_ch = base_channels
        layers += [
            _maybe_sn(
                nn.Conv2d(in_channels, out_ch, kernel_size=4, stride=2, padding=1, bias=True),
                use_spectral_norm,
            ),
            nn.LeakyReLU(negative_slope, inplace=True),
        ]

        # Downsampling blocks.
        in_ch = out_ch
        for i in range(1, n_downsamples):
            out_ch = min(base_channels * (2**i), max_channels)
            # If a norm follows, bias is unnecessary.
            bias = norm_kind.lower() in ("none", "identity")
            layers.append(
                _maybe_sn(
                    nn.Conv2d(in_ch, out_ch, kernel_size=4, stride=2, padding=1, bias=bias),
                    use_spectral_norm,
                )
            )
            layers.append(_norm_layer(norm_kind, out_ch))
            layers.append(nn.LeakyReLU(negative_slope, inplace=True))
            in_ch = out_ch

        self.features = nn.Sequential(*layers)
        self.pool = nn.AdaptiveAvgPool2d((4, 4)) if pool_to_4x4 else nn.Identity()

        self.use_conv_head = use_conv_head
        self.use_mlp_head = use_mlp_head

        self.conv_head = (
            _maybe_sn(nn.Conv2d(in_ch, 1, kernel_size=4, stride=1, padding=0, bias=True), use_spectral_norm)
            if use_conv_head
            else None
        )

        if use_mlp_head:
            mlp_layers: list[nn.Module] = []
            in_features = in_ch * 4 * 4
            prev = in_features
            for hidden in mlp_hidden:
                mlp_layers.append(_maybe_sn(nn.Linear(prev, hidden, bias=True), use_spectral_norm))
                mlp_layers.append(nn.LeakyReLU(negative_slope, inplace=True))
                if mlp_dropout > 0:
                    mlp_layers.append(nn.Dropout(mlp_dropout))
                prev = hidden
            mlp_layers.append(_maybe_sn(nn.Linear(prev, 1, bias=True), use_spectral_norm))
            self.mlp_head = nn.Sequential(*mlp_layers)
        else:
            self.mlp_head = None

        self.apply(self._init_weights)

    @staticmethod
    def _init_weights(module: nn.Module) -> None:
        if isinstance(module, (nn.Conv2d, nn.Linear)):
            nn.init.kaiming_normal_(module.weight, a=0.2, mode="fan_in", nonlinearity="leaky_relu")
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, (nn.BatchNorm2d, nn.InstanceNorm2d)):
            if getattr(module, "weight", None) is not None:
                nn.init.ones_(module.weight)
            if getattr(module, "bias", None) is not None:
                nn.init.zeros_(module.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feat = self.features(x)
        feat = self.pool(feat)

        logits: torch.Tensor | None = None

        if self.use_conv_head and self.conv_head is not None:
            conv_logits = self.conv_head(feat)  # (N, 1, 1, 1)
            logits = conv_logits if logits is None else (logits + conv_logits)

        if self.use_mlp_head and self.mlp_head is not None:
            mlp_logits = self.mlp_head(feat.flatten(1)).view(-1, 1, 1, 1)  # (N, 1, 1, 1)
            logits = mlp_logits if logits is None else (logits + mlp_logits)

        # logits can't be None due to init validation, but keep this defensive.
        if logits is None:
            raise RuntimeError("Discriminator has no active head")

        return logits


# Backward compatibility with the original (misspelled) name used in main.ipynb
Dicriminator = Discriminator