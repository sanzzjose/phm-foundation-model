"""Backbone PatchTST channel-independent adaptado al contrato PHM.

Inspirado en:

- PatchTST (Nie et al., 2023, https://arxiv.org/abs/2211.14730):
  patching temporal, channel-independence, embedding compartido por canal,
  Transformer encoder sobre tokens-patch. Aqui el patching ya se hizo en la
  fase de harmonization v0.5 (cada sample llega como `(C, N, P)` con N=32,
  P=16), por lo que el modelo no implementa el unfold; arranca directamente
  con el patch embedding sobre los patches ya construidos.

- MOMENT (Goswami et al., 2024, https://arxiv.org/abs/2402.03885):
  masked patch prediction como objetivo SSL principal sobre corpus
  multi-dataset heterogeneo. El mecanismo de `mask_token` aprendible y la
  cabeza de reconstruccion lineal estan tomados de ese enfoque.

Adaptaciones al contrato PHM:

- Channel-independence ESTRICTA: el patch embedding es `Linear(P, d_model)`,
  no `Linear(P*C, d_model)`. Ninguna capa inicial depende de `C`. La misma
  instancia debe aceptar C=1, C=24 y C=317 en llamadas distintas (lo
  imponen los tests).
- `valid_patch_mask` canonicalizado a `(B, C, N)` antes del forward.
- `src_key_padding_mask` del TransformerEncoder se construye con la
  validez de patches: patches *invalidos* se marcan como padding y se
  ignoran por la atencion. Patches *enmascarados pero validos* NO entran
  como padding, deben participar en la atencion (la prediccion fluye a
  traves de ellos).
- Positional embedding aprendible de longitud `N` fijo. Suficiente porque
  W=512 y patch_size=16 son constantes globales del pipeline (sec 12 de
  `CLAUDE.md`).
- La cabeza de reconstruccion es lineal `Linear(d_model, P)` aplicada por
  patch, manteniendo la simetria del esquema PatchTST/MOMENT.

Contrato del forward:

    Entrada:
      x:                (B, C, N, P)        float32
      valid_time_mask:  (B, W)              bool, W = N * P
      valid_patch_mask: aceptable cualquier shape canonicalizable a (B,C,N)
      ssl_mask:         (B, C, N)           bool. Si es None, no se sustituyen
                         embeddings por mask_token (modo "inferencia plana").

    Salida (dict):
      tokens:           (B, C, N, d_model)  float32
      reconstruction:   (B, C, N, P)        float32
      pooled:           (B, C, d_model)     float32 (mean pooling sobre
                         patches validos; cero si la fila no tiene validos)
"""

from __future__ import annotations

from typing import Optional, Tuple

import torch
import torch.nn as nn

from training.ssl.masking import (
    apply_mask_token,
    canonicalize_valid_patch_mask,
)


class PatchTSTPhm(nn.Module):
    """Encoder PatchTST channel-independent + cabeza de reconstruccion."""

    def __init__(
        self,
        patch_size: int = 16,
        n_patches: int = 32,
        d_model: int = 64,
        n_layers: int = 2,
        n_heads: int = 4,
        d_ff: int = 256,
        dropout: float = 0.1,
        activation: str = "gelu",
    ) -> None:
        super().__init__()
        # Estos valores son metadata (no parametros): se almacenan para que
        # los chequeos de shape sean explicitos y la trazabilidad sea total.
        self.patch_size = patch_size
        self.n_patches = n_patches
        self.d_model = d_model

        # 1. Patch embedding compartido (Linear(P, d_model)). Channel-independent:
        #    se aplica como una matmul sobre la ultima dimension P, igual para
        #    todos los canales y todos los batch elements.
        self.patch_embedding = nn.Linear(patch_size, d_model)

        # 2. Positional embedding learnable sobre las N posiciones de patch.
        #    No depende de C. Para el contrato MVP (W=512, P=16) → N=32.
        self.positional_embedding = nn.Parameter(
            torch.zeros(1, n_patches, d_model)
        )
        nn.init.trunc_normal_(self.positional_embedding, std=0.02)

        # 3. Mask token aprendible: parametro de shape (d_model,). Se sustituye
        #    en posiciones masked en espacio de embedding.
        self.mask_token = nn.Parameter(torch.zeros(d_model))
        nn.init.trunc_normal_(self.mask_token, std=0.02)

        # 4. Transformer encoder. batch_first=True para que el contrato sea
        #    (B*C, N, d_model). Norm pre-layer (norm_first=True) ayuda con
        #    estabilidad en MOMENT-style training.
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=d_ff,
            dropout=dropout,
            activation=activation,
            batch_first=True,
            norm_first=True,
        )
        # enable_nested_tensor=False silencia el UserWarning de PyTorch al
        # combinar norm_first=True con NestedTensor (la optimizacion no es
        # aplicable con pre-layer norm, que es la convencion MOMENT-style).
        self.encoder = nn.TransformerEncoder(
            encoder_layer, num_layers=n_layers, enable_nested_tensor=False,
        )
        self.encoder_norm = nn.LayerNorm(d_model)

        # 5. Cabeza de reconstruccion. Lineal por patch, simetrica al embedding.
        #    Linear(d_model, P) actua independiente por canal y por posicion.
        self.reconstruction_head = nn.Linear(d_model, patch_size)

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------

    def forward(
        self,
        x: torch.Tensor,
        valid_time_mask: torch.Tensor,
        valid_patch_mask: torch.Tensor,
        ssl_mask: Optional[torch.Tensor] = None,
    ) -> dict:
        """Forward pass channel-independent.

        Args:
            x: tensor `(B, C, N, P)` float32.
            valid_time_mask: tensor `(B, W)` bool, W = N*P.
            valid_patch_mask: tensor bool canonicalizable a `(B, C, N)`.
            ssl_mask: tensor bool `(B, C, N)` opcional. Si se pasa, los
                patches con `ssl_mask=True` ven su embedding sustituido por
                `mask_token` antes del transformer.

        Returns:
            Dict con `tokens`, `reconstruction`, `pooled` (ver docstring del
            modulo para shapes).
        """
        # ---------- Validacion basica ----------
        if x.dim() != 4:
            raise ValueError(f"x debe ser (B, C, N, P), recibido {tuple(x.shape)}")
        B, C, N, P = x.shape
        if P != self.patch_size:
            raise ValueError(
                f"patch_size del input ({P}) != configurado ({self.patch_size})"
            )
        if N != self.n_patches:
            raise ValueError(
                f"n_patches del input ({N}) != configurado ({self.n_patches})"
            )

        W = N * P
        if valid_time_mask.shape != (B, W) or valid_time_mask.dtype != torch.bool:
            raise ValueError(
                f"valid_time_mask debe ser bool (B={B}, W={W}), recibido "
                f"{tuple(valid_time_mask.shape)} dtype={valid_time_mask.dtype}"
            )

        # 1. Canonicalizar valid_patch_mask a (B, C, N)
        vpm_canon = canonicalize_valid_patch_mask(valid_patch_mask, B, C, N)

        # 2. Aplanar canales con el batch → (B*C, N, P)
        x_flat = x.reshape(B * C, N, P)

        # 3. Patch embedding compartido: Linear(P, d_model)
        emb = self.patch_embedding(x_flat)  # (B*C, N, d_model)

        # 4. Aplicar mask token a los patches enmascarados
        if ssl_mask is not None:
            if ssl_mask.shape != (B, C, N) or ssl_mask.dtype != torch.bool:
                raise ValueError(
                    f"ssl_mask debe ser bool (B={B}, C={C}, N={N}), recibido "
                    f"{tuple(ssl_mask.shape)} dtype={ssl_mask.dtype}"
                )
            ssl_flat = ssl_mask.reshape(B * C, N)
            emb = apply_mask_token(emb, ssl_flat, self.mask_token)

        # 5. Sumar positional embedding (broadcast sobre B*C)
        emb = emb + self.positional_embedding  # (B*C, N, d_model)

        # 6. Construir key_padding_mask para el transformer.
        #    Convencion de nn.TransformerEncoder: True donde IGNORAR.
        #    Marcamos como padding los patches con valid_patch_mask=False,
        #    NO los que estan en ssl_mask (esos si deben participar en la
        #    atencion para que la cabeza pueda reconstruirlos).
        vpm_flat = vpm_canon.reshape(B * C, N)
        key_padding_mask = ~vpm_flat  # True donde invalido

        # Edge case: si TODOS los patches de una fila (B*C) son padding,
        # nn.MultiheadAttention puede devolver NaN. Para protegerlo, dejamos
        # al menos un patch "visible" en la mascara aunque la fila no aporte
        # a la loss (la loss ya filtra por valid_time_mask).
        rows_all_pad = key_padding_mask.all(dim=1)
        if rows_all_pad.any():
            kp_safe = key_padding_mask.clone()
            kp_safe[rows_all_pad, 0] = False
            key_padding_mask = kp_safe

        # 7. Transformer encoder
        tokens_flat = self.encoder(emb, src_key_padding_mask=key_padding_mask)
        tokens_flat = self.encoder_norm(tokens_flat)  # (B*C, N, d_model)

        # 8. Cabeza de reconstruccion
        recon_flat = self.reconstruction_head(tokens_flat)  # (B*C, N, P)

        # 9. Reshape a (B, C, N, *)
        tokens = tokens_flat.reshape(B, C, N, self.d_model)
        recon = recon_flat.reshape(B, C, N, P)

        # 10. Pooling respetuoso con mascaras: media sobre patches validos.
        #     Si una fila no tiene validos, el pooled queda en cero.
        vpm_f = vpm_canon.to(tokens.dtype)  # (B, C, N)
        denom = vpm_f.sum(dim=2, keepdim=True).clamp(min=1.0)  # (B, C, 1)
        pooled = (tokens * vpm_f.unsqueeze(-1)).sum(dim=2) / denom  # (B, C, d_model)
        # Cero explicito en filas sin patches validos
        any_valid = vpm_canon.any(dim=2).to(tokens.dtype).unsqueeze(-1)
        pooled = pooled * any_valid

        return {
            "tokens": tokens,
            "reconstruction": recon,
            "pooled": pooled,
        }

    # ------------------------------------------------------------------
    # Convenience constructors
    # ------------------------------------------------------------------

    @classmethod
    def tiny(cls, patch_size: int = 16, n_patches: int = 32) -> "PatchTSTPhm":
        """Configuracion 'tiny' para tests y smoke.

        Aproximadamente **104.336 parametros entrenables**:
        Linear(P, d) + pos_emb (32, d) + mask_token (d) + 2 x TransformerLayer
        (d=64, n_heads=4, d_ff=256) + LayerNorm + Linear(d, P).
        Suficiente para validar el contrato end-to-end y para correr smoke
        en CPU/T4 sin problemas.
        """
        return cls(
            patch_size=patch_size,
            n_patches=n_patches,
            d_model=64,
            n_layers=2,
            n_heads=4,
            d_ff=256,
            dropout=0.1,
        )

    @classmethod
    def base(cls, patch_size: int = 16, n_patches: int = 32) -> "PatchTSTPhm":
        """Configuracion 'base' para pretraining real.

        Aproximadamente **801.808 parametros entrenables**:
        Linear(P, d) + pos_emb (32, d) + mask_token (d) + 4 x TransformerLayer
        (d=128, n_heads=4, d_ff=512) + LayerNorm + Linear(d, P).

        d_model=128 con 4 capas es un compromiso razonable de memoria para
        Colab A100/T4. Si se quiere subir d_model, n_layers o batch_size,
        justificar por VRAM disponible y por ratio compute/parametros.
        """
        return cls(
            patch_size=patch_size,
            n_patches=n_patches,
            d_model=128,
            n_layers=4,
            n_heads=4,
            d_ff=512,
            dropout=0.1,
        )


def count_parameters(model: "PatchTSTPhm", trainable_only: bool = True) -> int:
    """Cuenta parametros del modelo. Util para logging y trazabilidad.

    Args:
        model: instancia de `PatchTSTPhm`.
        trainable_only: si True, solo cuenta los que tienen `requires_grad`.
            Si False, cuenta todos (en este modelo no hay buffers grandes).
    """
    if trainable_only:
        return sum(p.numel() for p in model.parameters() if p.requires_grad)
    return sum(p.numel() for p in model.parameters())


def build_patchtst_phm(model_cfg: dict) -> PatchTSTPhm:
    """Construye un modelo desde un dict de configuracion YAML.

    Acepta los campos:
      d_model, n_layers, n_heads, d_ff, dropout, patch_size, n_patches.
    Si falta alguno, usa el default de `PatchTSTPhm`.
    """
    return PatchTSTPhm(
        patch_size=int(model_cfg.get("patch_size", 16)),
        n_patches=int(model_cfg.get("n_patches", 32)),
        d_model=int(model_cfg.get("d_model", 64)),
        n_layers=int(model_cfg.get("n_layers", 2)),
        n_heads=int(model_cfg.get("n_heads", 4)),
        d_ff=int(model_cfg.get("d_ff", 256)),
        dropout=float(model_cfg.get("dropout", 0.1)),
    )
