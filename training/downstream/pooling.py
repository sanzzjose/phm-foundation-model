"""Pooling de tokens PatchTST respetando mascaras de validez.

Contrato:

    tokens                    : (B, C, N, d_model)  salida del encoder
    valid_patch_mask          : (B, C, N)  bool, True donde el patch es real
    canales_constantes_mask   : (B, C)     bool, True donde el canal es
                                            constante (std cero o casi)

Estrategia de pooling channel-independent:

1) Reducir patches por canal:
       pooled_per_channel(b, c, :) = mean(tokens[b, c, i, :] for i in
                                          patches validos)
   Resultado: `(B, C, d_model)` con cero en `(b, c)` sin validos.

2) Reducir canales por sample, ignorando los constantes:
       pooled(b, :) = mean(pooled_per_channel[b, c, :] for c canal valido)
   Donde "canal valido" = NOT constante AND tiene >=1 patch real.

3) Si todos los canales del sample son invalidos o constantes,
   `pooled(b, :) = 0` (sin NaN).

Por que primero patches y luego canales:

- El encoder es channel-independent: cada canal pasa por el mismo encoder.
  El tensor de salida ya esta separado por canal.
- Promediar primero patches por canal evita que un canal con muchos
  patches validos domine sobre uno con pocos.
- Promediar despues canales es coherente con la idea de que cada canal es
  una "vista" del mismo asset.

Estabilidad numerica:

- Se usan denominadores `clamp(min=1)` para evitar division por cero.
- El producto con `valid_patch_mask.float()` y luego suma equivale al mean
  sobre los validos sin nans.
"""

from __future__ import annotations

from typing import Optional

import torch


def masked_patch_mean_pool(
    tokens: torch.Tensor, valid_patch_mask: torch.Tensor
) -> torch.Tensor:
    """Promedia tokens sobre patches validos por canal.

    Args:
        tokens:           `(B, C, N, d_model)` float.
        valid_patch_mask: `(B, C, N)` bool.

    Returns:
        `(B, C, d_model)` float. Filas `(b, c)` sin patches validos
        quedan en cero.
    """
    if tokens.dim() != 4:
        raise ValueError(f"tokens debe ser (B,C,N,d), recibido {tuple(tokens.shape)}")
    B, C, N, d = tokens.shape
    if valid_patch_mask.shape != (B, C, N):
        raise ValueError(
            f"valid_patch_mask.shape {tuple(valid_patch_mask.shape)} != ({B},{C},{N})"
        )
    if valid_patch_mask.dtype != torch.bool:
        raise ValueError(f"valid_patch_mask debe ser bool, recibido {valid_patch_mask.dtype}")
    m = valid_patch_mask.to(tokens.dtype)  # (B, C, N)
    # suma sobre N
    weighted = (tokens * m.unsqueeze(-1)).sum(dim=2)  # (B, C, d)
    denom = m.sum(dim=2, keepdim=True).clamp(min=1.0)  # (B, C, 1)
    pooled = weighted / denom
    # poner a cero filas (b,c) sin validos (cuando sum=0, denom=1 pero
    # tambien weighted=0; cero explicito para mayor robustez):
    any_valid = valid_patch_mask.any(dim=2)  # (B, C)
    pooled = pooled * any_valid.to(pooled.dtype).unsqueeze(-1)
    return pooled


def masked_channel_mean_pool(
    pooled_per_channel: torch.Tensor,
    channel_valid_mask: torch.Tensor,
) -> torch.Tensor:
    """Promedia canales validos por sample.

    Args:
        pooled_per_channel: `(B, C, d_model)` float, salida de
            `masked_patch_mean_pool`.
        channel_valid_mask: `(B, C)` bool. True donde el canal es valido
            (no constante y con >=1 patch real). El caller decide como
            combinar `valid_patch_mask.any(N)` con `~canales_constantes_mask`.

    Returns:
        `(B, d_model)` float. Filas `b` sin canales validos quedan en cero.
    """
    if pooled_per_channel.dim() != 3:
        raise ValueError(
            f"pooled_per_channel debe ser (B,C,d), recibido {tuple(pooled_per_channel.shape)}"
        )
    B, C, d = pooled_per_channel.shape
    if channel_valid_mask.shape != (B, C):
        raise ValueError(
            f"channel_valid_mask.shape {tuple(channel_valid_mask.shape)} != ({B},{C})"
        )
    if channel_valid_mask.dtype != torch.bool:
        raise ValueError(
            f"channel_valid_mask debe ser bool, recibido {channel_valid_mask.dtype}"
        )
    m = channel_valid_mask.to(pooled_per_channel.dtype)  # (B, C)
    weighted = (pooled_per_channel * m.unsqueeze(-1)).sum(dim=1)  # (B, d)
    denom = m.sum(dim=1, keepdim=True).clamp(min=1.0)  # (B, 1)
    pooled = weighted / denom
    any_valid = channel_valid_mask.any(dim=1).to(pooled.dtype).unsqueeze(-1)  # (B, 1)
    return pooled * any_valid


def pooled_embedding(
    tokens: torch.Tensor,
    valid_patch_mask: torch.Tensor,
    canales_constantes_mask: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Embedding por sample: patches por canal + canales por sample.

    Args:
        tokens:                  `(B, C, N, d_model)` float, salida encoder.
        valid_patch_mask:        `(B, C, N)` bool.
        canales_constantes_mask: `(B, C)` bool opcional. True donde el canal
            es constante (sera EXCLUIDO del pooling de canales). Si es None
            se asume que ningun canal es constante.

    Returns:
        `(B, d_model)` float. Cero si todo invalido/constante.
    """
    B, C, N, d = tokens.shape
    pooled_per_channel = masked_patch_mean_pool(tokens, valid_patch_mask)  # (B,C,d)

    # canal valido = al menos 1 patch real Y NO marcado como constante
    any_valid_patch = valid_patch_mask.any(dim=2)  # (B, C) bool
    if canales_constantes_mask is None:
        channel_valid = any_valid_patch
    else:
        if canales_constantes_mask.shape != (B, C):
            raise ValueError(
                f"canales_constantes_mask.shape "
                f"{tuple(canales_constantes_mask.shape)} != ({B},{C})"
            )
        if canales_constantes_mask.dtype != torch.bool:
            raise ValueError(
                f"canales_constantes_mask debe ser bool, recibido "
                f"{canales_constantes_mask.dtype}"
            )
        channel_valid = any_valid_patch & (~canales_constantes_mask)

    # Fallback: si TODOS los canales se descartan (todos constantes), no
    # tiene sentido devolver cero "ciego". Forzar al menos los canales
    # con patches validos aunque sean constantes (mejor que NaN/cero ciego).
    no_channel_left = ~channel_valid.any(dim=1)  # (B,) bool
    if no_channel_left.any():
        channel_valid = channel_valid | (
            no_channel_left.unsqueeze(-1) & any_valid_patch
        )

    return masked_channel_mean_pool(pooled_per_channel, channel_valid)
