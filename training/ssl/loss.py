"""Loss de reconstruccion para masked patch prediction (SSL).

La idea del SSL es: dado un batch de patches `(B, C, N, P)`, ocultar una
fraccion `mask_ratio` de patches validos (`valid_patch_mask`), pasar el resto
por el encoder y pedirle a la cabeza que reconstruya los patches ocultos. La
loss compara la reconstruccion con el patch original.

El detalle critico que motiva este modulo: con `tail_policy='pad'` activo en
toda la harmonizacion v0.5, cualquier trayectoria con `T > W` y resto > 0
produce una ventana parcial cuyo ultimo patch valido es "mitad real, mitad
padding". Si la loss usase solo `valid_patch_mask` (boolean a nivel de
patch), pediria al modelo predecir tambien los timesteps de padding, que son
artefactos. La cabeza aprenderia ruido.

La solucion correcta es usar tambien `valid_time_mask` (boolean a nivel de
timestep, shape `(W,)`), reshapearlo a `(N, P)` para alinear con la
estructura de patches, y aplicarlo elementwise dentro de cada patch
enmascarado.

Esto no se hace en la persistencia: las mascaras ya estan guardadas en cada
sample (sec 9 y 14 de `CLAUDE.md`). Aqui solo se *consume*.

Diseno y contrato (canonico, channel-independent):

    pred             : (B, C, N, P) float — salida de la cabeza
    target           : (B, C, N, P) float — patches originales
    ssl_mask         : (B, C, N)    bool  — True donde el patch fue ocultado
    valid_time_mask  : (B, W)       bool  — True donde el timestep es real.
                                            W = N * P obligatorio.

Notas:

- `valid_time_mask` es channel-independent: las mascaras valen para todos
  los canales de cada batch element. Si en el futuro se introduce
  enmascaramiento de canales muertos, anadir otra mascara `(B, C, N)` y
  combinar.
- `ssl_mask` debe ser un subconjunto de `valid_patch_mask` (no se enmascara
  un patch de puro padding). Esta funcion no lo asume: si por error llega un
  `ssl_mask` con patches invalidos, su contribucion sera cero por construccion.
- La funcion es estable bajo numero de elementos cero: si ningun timestep
  cumple `(ssl & valid)`, devuelve `0.` (y no `nan`) con `reduction='mean'`.

Convenciones:

- Por defecto, `loss_fn='mse'` y `reduction='mean'`, lo mismo que se usa en
  MOMENT / PatchTST para masked patch prediction.
- `reduction='none'` devuelve el tensor elementwise multiplicado por la
  mascara, util para inspeccion / debugging.
"""

from __future__ import annotations

from typing import Dict, Optional

import torch


def masked_reconstruction_loss(
    pred: torch.Tensor,
    target: torch.Tensor,
    ssl_mask: torch.Tensor,
    valid_time_mask: torch.Tensor,
    loss_fn: str = "mse",
    reduction: str = "mean",
) -> torch.Tensor:
    """Loss de reconstruccion SSL ignorando padding.

    Solo contribuyen al error los timesteps `(b, c, n, p)` tales que:
      1. `ssl_mask[b, c, n]` (el patch fue ocultado para SSL), Y
      2. `valid_time_mask[b, n*P + p]` (el timestep es real, no padding).

    Args:
        pred: tensor `(B, C, N, P)` con la reconstruccion del modelo.
        target: tensor `(B, C, N, P)` con los patches originales.
        ssl_mask: tensor bool `(B, C, N)`. True donde el patch esta ocultado.
        valid_time_mask: tensor bool `(B, W)` con W = N * P. True donde el
            timestep es real.
        loss_fn: `"mse"` (cuadratica) o `"mae"` (absoluta).
        reduction: `"mean"` (default), `"sum"` o `"none"`.

    Returns:
        Si `reduction='mean'` o `'sum'`: escalar `()`. Si `reduction='none'`:
        tensor `(B, C, N, P)` con los errores elementwise ya enmascarados
        (todo lo no contribuyente esta en 0).

    Raises:
        ValueError si las shapes no son consistentes.
    """
    # ---------- Validacion de shapes ----------
    if pred.shape != target.shape:
        raise ValueError(
            f"pred.shape ({tuple(pred.shape)}) != target.shape ({tuple(target.shape)})"
        )
    if pred.dim() != 4:
        raise ValueError(
            f"pred debe ser (B, C, N, P), recibido shape {tuple(pred.shape)}"
        )
    B, C, N, P = pred.shape
    W = N * P

    if ssl_mask.shape != (B, C, N):
        raise ValueError(
            f"ssl_mask.shape ({tuple(ssl_mask.shape)}) != esperado (B={B}, C={C}, N={N})"
        )
    if valid_time_mask.shape != (B, W):
        raise ValueError(
            f"valid_time_mask.shape ({tuple(valid_time_mask.shape)}) != "
            f"esperado (B={B}, W={W} = N*P)"
        )
    if ssl_mask.dtype != torch.bool:
        raise ValueError(f"ssl_mask debe ser bool, recibido dtype={ssl_mask.dtype}")
    if valid_time_mask.dtype != torch.bool:
        raise ValueError(
            f"valid_time_mask debe ser bool, recibido dtype={valid_time_mask.dtype}"
        )

    if loss_fn not in ("mse", "mae"):
        raise ValueError(f"loss_fn debe ser 'mse' o 'mae', recibido {loss_fn!r}")
    if reduction not in ("mean", "sum", "none"):
        raise ValueError(
            f"reduction debe ser 'mean', 'sum' o 'none', recibido {reduction!r}"
        )

    # ---------- Construir la mascara elementwise (B, C, N, P) bool ----------
    # 1. valid_time_mask se reshapea de (B, W) a (B, 1, N, P) y se expande a C.
    #    Es channel-independent: los mismos timesteps valen para todos los canales.
    # Usamos .reshape en lugar de .view: .view exige contiguidad estricta y
    # falla si el tensor entrante viene de un slicing/transpose no contiguo.
    # .reshape se comporta igual cuando es contiguo y hace una copia barata
    # cuando no lo es. La semantica de la loss no cambia.
    valid_sample = valid_time_mask.reshape(B, 1, N, P).expand(B, C, N, P)
    # 2. ssl_mask es (B, C, N) → (B, C, N, 1) para broadcast sobre P.
    ssl_expanded = ssl_mask.unsqueeze(-1)
    # 3. La mascara final es la AND elementwise. Donde es False, el error se
    #    multiplica por 0 y no contribuye.
    contrib = valid_sample & ssl_expanded  # (B, C, N, P) bool

    # ---------- Error elementwise sin reduccion ----------
    if loss_fn == "mse":
        err = (pred - target).pow(2)
    else:  # mae
        err = (pred - target).abs()

    masked_err = err * contrib.to(err.dtype)

    # ---------- Reduccion ----------
    if reduction == "none":
        return masked_err
    if reduction == "sum":
        return masked_err.sum()
    # 'mean': divide por numero de elementos que contribuyen (no por B*C*N*P).
    # Si nada contribuye (caso degenerado), devuelve 0 sin nan.
    n_contrib = contrib.sum().clamp(min=1)
    return masked_err.sum() / n_contrib.to(masked_err.dtype)


def derivar_valid_sample_mask(
    valid_time_mask: torch.Tensor, n_patches: int, patch_size: int
) -> torch.Tensor:
    """Deriva la mascara timestep-wise reshapeada (B, N, P).

    Helper independiente para introspeccion. La funcion `masked_reconstruction_loss`
    la calcula internamente expandida a (B, C, N, P), pero a veces se necesita
    `(B, N, P)` para diagnostico (e.g. visualizar que patches son parciales).

    Args:
        valid_time_mask: tensor bool `(B, W)` con W = n_patches * patch_size.
        n_patches: numero de patches por ventana (`N`).
        patch_size: tamano de patch en samples (`P`).

    Returns:
        Tensor bool `(B, N, P)`. `valid_sample_mask[b, n, p]` == True si el
        timestep `n*P + p` es real en el batch element `b`.

    Raises:
        ValueError si las shapes no encajan.
    """
    if valid_time_mask.dim() != 2:
        raise ValueError(
            f"valid_time_mask debe ser (B, W), recibido {tuple(valid_time_mask.shape)}"
        )
    B, W = valid_time_mask.shape
    if W != n_patches * patch_size:
        raise ValueError(
            f"W={W} != N*P = {n_patches}*{patch_size} = {n_patches * patch_size}"
        )
    if valid_time_mask.dtype != torch.bool:
        raise ValueError(f"valid_time_mask debe ser bool, recibido {valid_time_mask.dtype}")
    # .reshape para robustez ante tensores no contiguos; ver nota en
    # masked_reconstruction_loss.
    return valid_time_mask.reshape(B, n_patches, patch_size)


def compute_masked_reconstruction_loss_with_metrics(
    pred: torch.Tensor,
    target: torch.Tensor,
    ssl_mask: torch.Tensor,
    valid_time_mask: torch.Tensor,
    valid_patch_mask: Optional[torch.Tensor] = None,
    loss_fn: str = "mse",
) -> Dict[str, torch.Tensor]:
    """Wrapper que devuelve loss + metricas escalares para el loop SSL.

    Reutiliza `masked_reconstruction_loss` como nucleo (no duplica la
    implementacion). El wrapper anade dos cosas:

    1. **`valid_patch_mask` opcional**: si se pasa, se combina con la mascara
       final `(ssl & valid_time)` para imponer ademas que el patch sea
       valido como entidad. En la harmonization v0.5, `ssl_mask` ya se
       construye solo sobre patches validos (sec `generate_ssl_mask`), asi
       que `valid_patch_mask` es redundante por contrato; lo aceptamos como
       red de seguridad para experimentos donde se inyecte un `ssl_mask`
       externo.

    2. **Metricas escalares para logging**:
       - `loss`: tensor escalar (default MSE), promediado sobre los
         elementos que contribuyen.
       - `n_loss_elements`: int, numero de timesteps que entran en la loss.
       - `n_masked_patches`: int, numero total de patches enmascarados (sin
         considerar validez fina).
       - `effective_mask_ratio`: float, fraccion masked-valid / total-valid.
       - `padding_ignored_elements`: int, timesteps que el wrapper *ignora*
         por estar en padding dentro de patches enmascarados.
       - `n_valid_patches`: int, numero de patches validos en el batch.

    Si no hay ningun elemento que contribuya, devuelve `loss=0` (gracias al
    `clamp(min=1)` del nucleo) y `n_loss_elements=0`. El loop de training
    debe saltar batches con `n_loss_elements == 0`.

    Args:
        pred: tensor `(B, C, N, P)` con la reconstruccion del modelo.
        target: tensor `(B, C, N, P)` con los patches originales.
        ssl_mask: tensor bool `(B, C, N)`. True donde el patch esta oculto.
        valid_time_mask: tensor bool `(B, W)` con W = N * P.
        valid_patch_mask: tensor bool opcional `(B, C, N)` con la validez
            de patches a nivel de entidad. Si se pasa, se aplica como AND
            adicional. Si es None, no se usa.
        loss_fn: "mse" o "mae".

    Returns:
        Dict con keys: `loss`, `n_loss_elements`, `n_masked_patches`,
        `effective_mask_ratio`, `padding_ignored_elements`, `n_valid_patches`.
        Todos los valores son tensores escalares.

    Raises:
        ValueError si las shapes no son consistentes.
    """
    if pred.shape != target.shape or pred.dim() != 4:
        raise ValueError(
            f"pred y target deben ser (B,C,N,P) y coincidir: "
            f"pred={tuple(pred.shape)}, target={tuple(target.shape)}"
        )
    B, C, N, P = pred.shape
    W = N * P
    if ssl_mask.shape != (B, C, N) or ssl_mask.dtype != torch.bool:
        raise ValueError(
            f"ssl_mask debe ser bool (B={B}, C={C}, N={N}), recibido "
            f"{tuple(ssl_mask.shape)} dtype={ssl_mask.dtype}"
        )
    if valid_time_mask.shape != (B, W) or valid_time_mask.dtype != torch.bool:
        raise ValueError(
            f"valid_time_mask debe ser bool (B={B}, W={W}), recibido "
            f"{tuple(valid_time_mask.shape)} dtype={valid_time_mask.dtype}"
        )

    # Si nos pasan valid_patch_mask, debe ser (B, C, N) bool. Lo combinamos
    # como red de seguridad con ssl_mask antes de pasar al nucleo.
    if valid_patch_mask is not None:
        if (
            valid_patch_mask.shape != (B, C, N)
            or valid_patch_mask.dtype != torch.bool
        ):
            raise ValueError(
                f"valid_patch_mask debe ser bool (B={B}, C={C}, N={N}), "
                f"recibido {tuple(valid_patch_mask.shape)} dtype={valid_patch_mask.dtype}"
            )
        ssl_effective = ssl_mask & valid_patch_mask
    else:
        ssl_effective = ssl_mask

    # Nucleo: la loss reducida (mean) ya divide por contribuyentes reales.
    loss = masked_reconstruction_loss(
        pred=pred,
        target=target,
        ssl_mask=ssl_effective,
        valid_time_mask=valid_time_mask,
        loss_fn=loss_fn,
        reduction="mean",
    )

    # Para metricas necesitamos los conteos exactos de la mascara final.
    valid_sample = valid_time_mask.reshape(B, 1, N, P).expand(B, C, N, P)
    loss_contrib = ssl_effective.unsqueeze(-1) & valid_sample  # (B, C, N, P)
    n_loss_elements = int(loss_contrib.sum().item())

    # Cuantos timesteps de padding fueron "salvados" dentro de patches
    # enmascarados (relevantes para tail_policy=pad).
    masked_all_timesteps = ssl_effective.unsqueeze(-1).expand(B, C, N, P)
    padding_ignored = int((masked_all_timesteps & ~valid_sample).sum().item())

    n_masked_patches = int(ssl_effective.sum().item())
    n_valid_patches = (
        int(valid_patch_mask.sum().item())
        if valid_patch_mask is not None
        else -1  # no calculable sin la mascara de validez de patches
    )
    if n_valid_patches > 0:
        effective_mask_ratio = float(n_masked_patches / n_valid_patches)
    else:
        effective_mask_ratio = 0.0

    return {
        "loss": loss,
        "n_loss_elements": torch.tensor(n_loss_elements),
        "n_masked_patches": torch.tensor(n_masked_patches),
        "effective_mask_ratio": torch.tensor(effective_mask_ratio),
        "padding_ignored_elements": torch.tensor(padding_ignored),
        "n_valid_patches": torch.tensor(n_valid_patches),
    }
