import equinox as eqx
import jax.numpy as jnp
from jax import random, vmap
from jax.nn import gelu
from jax.lax import dot_general, scan, dynamic_slice_in_dim
from jax.tree_util import tree_map
from transforms import utilities  # ваш модуль утилит


# —————————————————————————————————————————————
# Вспомогательная spectral‐conv (без изменений)
# —————————————————————————————————————————————
class SpectralConv1d(eqx.Module):
    weight: jnp.ndarray  # complex weights [in,out,modes]
    bias: jnp.ndarray  # real bias [out]
    modes: int

    def __init__(self, in_ch, out_ch, modes, key):
        k1, k2 = random.split(key)
        scale = 1 / jnp.sqrt(in_ch * out_ch)
        self.weight = (
            random.normal(k1, (in_ch, out_ch, modes), dtype=jnp.complex64) * scale
        )
        self.bias = jnp.zeros((out_ch,))
        self.modes = modes

    def __call__(self, x):
        # x: (B, C_in, L)
        x_ft = jnp.fft.rfft(x, axis=-1)
        x_low = x_ft[:, :, : self.modes]
        # спектральная свёртка
        y_ft = jnp.einsum("bim, iom -> bom", x_low, self.weight)
        # обратное преобразование
        y = jnp.fft.irfft(y_ft, n=x.shape[-1], axis=-1)
        return y + self.bias[None, :, None]


# —————————————————————————————————————————————
# Multi‐scale Fourier Neural Operator
# —————————————————————————————————————————————
class MSFNO1d(eqx.Module):
    encoder: eqx.Module
    decoder: eqx.Module
    convs1: list
    convs2: list
    specs: jnp.ndarray  # [layers, scales]
    modes_list: list  # e.g. [L//8, L//4, L//2]

    def __init__(self, N_layers, N_features, grid_size, modes_list, key):
        """
        N_layers   — число слоёв spectral‐conv
        N_features — кортеж (n_in, n_hidden, n_out)
        grid_size  — длина сетки (L)
        modes_list — список масштабов spectral‐conv, например [L//8, L//4, L//2]
        """
        n_in, n_hidden, n_out = N_features
        keys = random.split(key, 3 + 2 * N_layers)
        # 1×1 свёртки
        self.encoder = utilities.normalize_conv(
            eqx.nn.Conv1d(n_in, n_hidden, 1, key=keys[-1])
        )
        self.decoder = utilities.normalize_conv(
            eqx.nn.Conv1d(n_hidden, n_out, 1, key=keys[-2])
        )
        # списки pointwise‐свёрток
        self.convs1 = [
            utilities.normalize_conv(eqx.nn.Conv1d(n_hidden, n_hidden, 1, key=k))
            for k in keys[:N_layers]
        ]
        self.convs2 = [
            utilities.normalize_conv(eqx.nn.Conv1d(n_hidden, n_hidden, 1, key=k))
            for k in keys[N_layers : 2 * N_layers]
        ]
        # сохраняем список масштабов
        self.modes_list = modes_list

    def __call__(self, u, x):
        """
        u — (B, C_in, L), x — координаты (например, для позкод)
        """
        # кодирование
        v = jnp.concatenate([x, u], axis=1)  # (B, n_in, L)
        v = self.encoder(v)  # (B, n_hidden, L)

        # N_layers блоков
        for conv1, conv2 in zip(self.convs1, self.convs2):
            # суммируем несколько спектральных конволюций
            specs = [
                SpectralConv1d(v.shape[1], v.shape[1], m, key=None)(v)
                for m in self.modes_list
            ]
            spec_sum = sum(specs)  # (B, n_hidden, L)
            # pointwise + GELU
            delta = conv2(gelu(conv1(spec_sum)))
            v = v + delta

        # декодирование
        out = self.decoder(v)  # (B, n_out, L)
        return out
