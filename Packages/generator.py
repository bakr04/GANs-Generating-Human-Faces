import math

import torch
import torch.nn as nn


def _num_upsamples(img_size: int, init_size: int) -> int:
    if img_size % init_size != 0:
        raise ValueError("img_size must be divisible by init_size")
    ratio = img_size // init_size
    if ratio & (ratio - 1) != 0:
        raise ValueError("img_size/init_size must be a power of two")
    return int(math.log2(ratio))


class _ResUpsampleBlock(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, negative_slope: float = 0.2):
        super().__init__()
        self.upsample = nn.Upsample(scale_factor=2, mode="nearest")

        self.conv1 = nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_ch)
        self.act = nn.LeakyReLU(negative_slope, inplace=True)

        self.conv2 = nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_ch)

        self.skip = nn.Conv2d(in_ch, out_ch, kernel_size=1, bias=False) if in_ch != out_ch else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_up = self.upsample(x)
        y = self.conv1(x_up)
        y = self.bn1(y)
        y = self.act(y)
        y = self.conv2(y)
        y = self.bn2(y)
        y = y + self.skip(x_up)
        y = self.act(y)
        return y


class Generator(nn.Module):
    """Generator with mapping MLP + CNN upsampling.

    Defaults keep the original behavior:
    - z_dim=128
    - output: (N, 1, 256, 256) in [-1, 1]
    """

    def __init__(
        self,
        z_dim: int = 128,
        out_channels: int = 1,
        img_size: int = 256,
        init_size: int = 4,
        init_channels: int = 512,
        channel_schedule: tuple[int, ...] | None = (512, 256, 128, 64, 32, 16),
        w_dim: int = 256,
        mapping_layers: int = 2,
        mapping_hidden: int = 256,
        negative_slope: float = 0.2,
        output_mode: str = "tanh",
        dataset_mean: float = 0.5,
        dataset_std: float = 0.225,
    ):
        super().__init__()

        if z_dim <= 0:
            raise ValueError("z_dim must be > 0")
        if out_channels <= 0:
            raise ValueError("out_channels must be > 0")
        if init_size <= 0:
            raise ValueError("init_size must be > 0")
        if init_channels <= 0:
            raise ValueError("init_channels must be > 0")

        self.z_dim = z_dim
        self.img_size = img_size
        self.init_size = init_size
        self.init_channels = init_channels
        self.output_mode = output_mode
        self.dataset_mean = float(dataset_mean)
        self.dataset_std = float(dataset_std)

        if self.dataset_std <= 0:
            raise ValueError("dataset_std must be > 0")

        # Mapping network (MLP) before projecting to spatial features.
        mlp: list[nn.Module] = []
        in_f = z_dim
        for i in range(mapping_layers):
            out_f = w_dim if i == mapping_layers - 1 else mapping_hidden
            mlp.append(nn.Linear(in_f, out_f))
            if i != mapping_layers - 1:
                mlp.append(nn.LeakyReLU(negative_slope, inplace=True))
            in_f = out_f
        self.mapping = nn.Sequential(*mlp) if mapping_layers > 0 else nn.Identity()

        self.fc = nn.Linear(w_dim if mapping_layers > 0 else z_dim, init_channels * init_size * init_size)

        # Upsampling trunk.
        n_up = _num_upsamples(img_size=img_size, init_size=init_size)
        if channel_schedule is None:
            # Simple fallback schedule if not provided.
            channel_schedule = tuple(max(16, init_channels // (2**i)) for i in range(1, n_up + 1))

        if len(channel_schedule) != n_up:
            raise ValueError(
                f"channel_schedule must have length {n_up} for img_size={img_size} and init_size={init_size}"
            )

        blocks: list[nn.Module] = []
        in_ch = init_channels
        for out_ch in channel_schedule:
            blocks.append(_ResUpsampleBlock(in_ch, out_ch, negative_slope=negative_slope))
            in_ch = out_ch
        self.trunk = nn.Sequential(*blocks)

        self.to_image = nn.Conv2d(in_ch, out_channels, kernel_size=3, padding=1)

        self.apply(self._init_weights)

    @staticmethod
    def _init_weights(module: nn.Module) -> None:
        if isinstance(module, (nn.Conv2d, nn.Linear)):
            nn.init.kaiming_normal_(module.weight, a=0.2, mode="fan_in", nonlinearity="leaky_relu")
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.BatchNorm2d):
            nn.init.ones_(module.weight)
            nn.init.zeros_(module.bias)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        w = self.mapping(z)
        x = self.fc(w)
        x = x.view(z.size(0), self.init_channels, self.init_size, self.init_size)
        x = self.trunk(x)
        x = self.to_image(x)

        mode = self.output_mode.lower()
        if mode in ("tanh",):
            return torch.tanh(x)
        if mode in ("sigmoid", "pixel"):
            return torch.sigmoid(x)
        if mode in ("dataset", "dataset_norm", "normalized"):
            # Match the dataset space used by GrayscaleImageDataset:
            # transforms.Normalize(mean=[dataset_mean], std=[dataset_std])
            # Generate pixels in [0,1] then normalize.
            pixels = torch.sigmoid(x)
            return (pixels - self.dataset_mean) / self.dataset_std

        raise ValueError(f"Unknown output_mode: {self.output_mode!r}")
