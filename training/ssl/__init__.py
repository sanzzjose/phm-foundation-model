"""Modulos para Self-Supervised Learning (masked patch prediction).

El SSL se entrena sobre los shards harmonizados del corpus PS (audit v2.3 +
full v0.5). El contrato tensorial es channel-independent: cada canal pasa
por el encoder con los mismos pesos.

Por dataset, cada sample del shard contiene:

    patches            : (C, N, P) float32, con padding implicito al final
    valid_time_mask    : (W,) bool, W = N * P, True donde hay senal real
    valid_patch_mask   : (N,) bool, True si el patch tiene al menos un
                          timestep real

La loss SSL debe ignorar el padding *dentro de patches parcialmente
validos*. Esto se vuelve obligatorio con `tail_policy='pad'` activo, donde
toda trayectoria con T > W y resto > 0 genera una ventana parcial cuyo
ultimo patch valido es mitad real / mitad padding. Sin la mascara fina, la
cabeza de reconstruccion aprende ruido sobre las muestras de padding.

Modulos:

- `loss.masked_reconstruction_loss`: implementacion canonica de la loss SSL
  channel-independent que combina `ssl_mask` (que patches estan ocultos
  para la prediccion) con `valid_time_mask` (que timesteps son reales). Ver
  docstring del modulo para los contratos exactos y casos limite.
"""

from training.ssl.loss import (
    compute_masked_reconstruction_loss_with_metrics,
    masked_reconstruction_loss,
)
from training.ssl.masking import (
    apply_mask_token,
    canonicalize_valid_patch_mask,
    compute_effective_mask_ratio,
    generate_ssl_mask,
)

__all__ = [
    "masked_reconstruction_loss",
    "compute_masked_reconstruction_loss_with_metrics",
    "canonicalize_valid_patch_mask",
    "generate_ssl_mask",
    "apply_mask_token",
    "compute_effective_mask_ratio",
]
