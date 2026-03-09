"""Politicas de masking para masked patch prediction (SSL).

Tres funciones canonicas:

- `canonicalize_valid_patch_mask`: lleva la mascara de validez de patches a
  la forma estandar `(B, C, N)`. En disco se guarda como `(C, N)` por
  sample, despues del collate suele venir `(B, C, N)`, pero a veces (cuando
  se broadcasta) llega como `(B, N)` o `(B, 1, N)`. Esta funcion centraliza
  esa canonicalizacion.

- `generate_ssl_mask`: para cada `(b, c)`, selecciona aleatoriamente
  `round(mask_ratio * n_valid_b_c)` patches *de entre los validos*. Nunca
  selecciona patches invalidos. Garantiza `min_masks` cuando hay suficientes
  patches validos. Es la rutina canonica de masking dinamico por batch.

- `apply_mask_token`: reemplaza el embedding de los patches enmascarados
  (en espacio `(B*C, N, d_model)`) por un `mask_token` aprendible. La logica
  de "input al transformer con tokens visibles + mask tokens" vive aqui.

Decisiones de diseno:

1. Channel-independent: aunque podriamos elegir el mismo subset de patches
   para todos los canales de un sample, optamos por mascarado *por canal*.
   Esto es coherente con el resto del pipeline (cada canal pasa por el
   encoder de forma independiente) y con MOMENT.
2. Determinismo controlado: aceptamos un `generator` opcional para tests.
   Si no se pasa, se usa el RNG global de torch.
3. Edge cases:
   - `n_valid == 0`: la fila `(b, c)` no aporta nada a la loss, no
     enmascaramos nada y devolvemos `False` para todos sus patches.
   - `n_valid < min_masks`: enmascaramos todos los validos (la loss sigue
     siendo valida porque se promedia sobre contribuyentes reales).
   - `n_valid >= min_masks`: enmascaramos `max(min_masks, round(ratio * n_valid))`
     patches *entre los validos*.
"""

from __future__ import annotations

from typing import Optional

import torch


def canonicalize_valid_patch_mask(
    valid_patch_mask: torch.Tensor,
    B: int,
    C: int,
    N: int,
) -> torch.Tensor:
    """Lleva `valid_patch_mask` a la forma canonica `(B, C, N)` bool.

    Acepta:
        - `(B, C, N)`: passthrough (despues de validar tipo).
        - `(B, 1, N)`: expand a C.
        - `(B, N)`: unsqueeze + expand a C.
        - `(C, N)`: caso por-sample antes del collate, broadcast a B.
        - `(N,)`: caso aun mas reducido, broadcast a (B, C).

    Args:
        valid_patch_mask: tensor bool con cualquiera de las shapes aceptadas.
        B, C, N: dimensiones objetivo.

    Returns:
        Tensor `(B, C, N)` bool contiguo (copia barata via `.contiguous()`
        para no devolver vistas con strides que rompen `view` aguas abajo).

    Raises:
        ValueError si dtype no es bool o si la shape es incompatible.
    """
    if valid_patch_mask.dtype != torch.bool:
        raise ValueError(
            f"valid_patch_mask debe ser bool, recibido {valid_patch_mask.dtype}"
        )
    shape = tuple(valid_patch_mask.shape)

    if shape == (B, C, N):
        out = valid_patch_mask
    elif shape == (B, 1, N):
        out = valid_patch_mask.expand(B, C, N)
    elif shape == (B, N):
        out = valid_patch_mask.unsqueeze(1).expand(B, C, N)
    elif shape == (C, N):
        out = valid_patch_mask.unsqueeze(0).expand(B, C, N)
    elif shape == (N,):
        out = valid_patch_mask.view(1, 1, N).expand(B, C, N)
    else:
        raise ValueError(
            f"valid_patch_mask.shape={shape} incompatible con (B={B}, C={C}, N={N})"
        )
    return out.contiguous()


def generate_ssl_mask(
    valid_patch_mask: torch.Tensor,
    mask_ratio: float = 0.3,
    generator: Optional[torch.Generator] = None,
    min_masks: int = 1,
) -> torch.Tensor:
    """Genera `ssl_mask` dinamica para masked patch prediction.

    Para cada fila `(b, c)`:
      - obtiene los indices de patches validos;
      - calcula `n_target = max(min_masks, round(mask_ratio * n_valid))`,
        recortado por `n_valid`;
      - elige `n_target` indices al azar entre los validos.

    Args:
        valid_patch_mask: tensor bool `(B, C, N)`. Debe llegar canonicalizado.
        mask_ratio: fraccion de patches validos a enmascarar (por canal).
        generator: `torch.Generator` opcional para reproducibilidad.
        min_masks: minimo absoluto de patches enmascarados por `(b, c)`
            siempre que `n_valid >= 1`.

    Returns:
        Tensor `(B, C, N)` bool. `True` donde el patch fue elegido para SSL.
        Solo selecciona patches con `valid_patch_mask == True`.

    Raises:
        ValueError si dtype/shape mal o ratio fuera de `[0, 1]`.
    """
    if valid_patch_mask.dtype != torch.bool:
        raise ValueError(
            f"valid_patch_mask debe ser bool, recibido {valid_patch_mask.dtype}"
        )
    if valid_patch_mask.dim() != 3:
        raise ValueError(
            f"valid_patch_mask debe ser (B, C, N), recibido {tuple(valid_patch_mask.shape)}"
        )
    if not (0.0 <= mask_ratio <= 1.0):
        raise ValueError(f"mask_ratio debe estar en [0, 1], recibido {mask_ratio}")
    if min_masks < 0:
        raise ValueError(f"min_masks debe ser >= 0, recibido {min_masks}")

    B, C, N = valid_patch_mask.shape
    device = valid_patch_mask.device
    ssl_mask = torch.zeros((B, C, N), dtype=torch.bool, device=device)

    # Recorremos (b, c). El bucle Python es aceptable para B*C tipico (32-256);
    # vectorizar con scatter es posible pero menos legible para tests.
    for b in range(B):
        for c in range(C):
            row = valid_patch_mask[b, c]
            n_valid = int(row.sum().item())
            if n_valid == 0:
                continue
            # Numero objetivo de mascaras dentro de los validos
            n_target = max(min_masks, int(round(mask_ratio * n_valid)))
            n_target = min(n_target, n_valid)
            # Seleccion aleatoria de n_target indices entre los validos
            valid_idx = torch.nonzero(row, as_tuple=False).squeeze(-1)
            if generator is not None:
                perm = torch.randperm(n_valid, generator=generator, device=device)
            else:
                perm = torch.randperm(n_valid, device=device)
            chosen = valid_idx[perm[:n_target]]
            ssl_mask[b, c, chosen] = True
    return ssl_mask


def apply_mask_token(
    embeddings: torch.Tensor,
    ssl_mask_flat: torch.Tensor,
    mask_token: torch.Tensor,
) -> torch.Tensor:
    """Sustituye los embeddings de patches enmascarados por `mask_token`.

    Args:
        embeddings: tensor `(B*C, N, d_model)`. La dimension de canales ya
            esta aplanada con el batch para channel-independence.
        ssl_mask_flat: tensor bool `(B*C, N)`. True donde se sustituye.
        mask_token: parametro aprendible de shape `(d_model,)` o `(1, 1, d_model)`.

    Returns:
        Tensor `(B*C, N, d_model)` con los patches enmascarados sustituidos.
        Devuelve un tensor nuevo: NO modifica `embeddings` in-place.
    """
    if embeddings.dim() != 3:
        raise ValueError(
            f"embeddings debe ser (B*C, N, d_model), recibido {tuple(embeddings.shape)}"
        )
    BC, N, d_model = embeddings.shape
    if ssl_mask_flat.shape != (BC, N):
        raise ValueError(
            f"ssl_mask_flat.shape={tuple(ssl_mask_flat.shape)} != esperado ({BC}, {N})"
        )
    if ssl_mask_flat.dtype != torch.bool:
        raise ValueError(
            f"ssl_mask_flat debe ser bool, recibido {ssl_mask_flat.dtype}"
        )
    mask_tok = mask_token.view(1, 1, d_model).to(embeddings.dtype)
    # broadcast (BC, N, 1) AND (1, 1, d_model) → reemplazo elementwise
    mask_4d = ssl_mask_flat.unsqueeze(-1)  # (BC, N, 1)
    return torch.where(mask_4d, mask_tok.expand_as(embeddings), embeddings)


def compute_effective_mask_ratio(
    ssl_mask: torch.Tensor, valid_patch_mask: torch.Tensor
) -> float:
    """Calcula el ratio efectivo: `masked_validos / total_validos`.

    Util para logging y validacion: en presencia de muchas filas con pocos
    `n_valid`, el ratio puntual por fila puede no llegar al target, pero el
    promedio sobre el batch sigue siendo informativo.
    """
    if ssl_mask.shape != valid_patch_mask.shape:
        raise ValueError(
            f"ssl_mask.shape={tuple(ssl_mask.shape)} != "
            f"valid_patch_mask.shape={tuple(valid_patch_mask.shape)}"
        )
    n_valid = int(valid_patch_mask.sum().item())
    if n_valid == 0:
        return 0.0
    n_masked = int((ssl_mask & valid_patch_mask).sum().item())
    return n_masked / n_valid
