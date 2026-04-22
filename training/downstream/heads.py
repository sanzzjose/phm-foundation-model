"""Cabezas downstream sobre el embedding pooled del PatchTSTPhm.

Cabezas disponibles:
- `ClassificationHead` + `DownstreamClassifier` (classification_multiclass).
- `RegressionHead` + `RegressionDownstreamModel` (regresion escalar,
  e.g. RUL en CMAPSS).

Los dos wrappers comparten contrato: backbone PatchTSTPhm + pooling +
cabeza. Soportan dos modos via `freeze_backbone`:

- `freeze_backbone=False`: el backbone es entrenable (modos `from_scratch`
  y `full_finetuning` del trainer).
- `freeze_backbone=True`: backbone congelado (modo `linear_probing`).

Convencion: la cabeza siempre recibe el embedding pooled `(B, d_model)`,
nunca tokens crudos `(B, C, N, d_model)`. El pooling vive en
`training.downstream.pooling`. Asi las cabezas no dependen del numero de
canales C del dataset, manteniendo la propiedad channel-independent
heredada del encoder.
"""

from __future__ import annotations

from typing import Any, Dict, Optional

import torch
import torch.nn as nn

from training.downstream.pooling import pooled_embedding


class ClassificationHead(nn.Module):
    """Cabeza lineal: `Linear(d_model, n_classes)`.

    Es deliberadamente minima. Para experimentos con regularizacion mas
    fuerte (dropout, LayerNorm en la cabeza, MLP de dos capas) se puede
    sustituir esta capa por una version mas amplia sin tocar el resto del
    pipeline downstream.
    """

    def __init__(self, d_model: int, n_classes: int, dropout: float = 0.0):
        super().__init__()
        if n_classes < 2:
            raise ValueError(f"n_classes debe ser >=2, recibido {n_classes}")
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        self.fc = nn.Linear(d_model, n_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Args:
            x: `(B, d_model)` embedding pooled.
        Returns:
            `(B, n_classes)` logits.
        """
        if x.dim() != 2:
            raise ValueError(f"x debe ser (B, d_model), recibido {tuple(x.shape)}")
        return self.fc(self.dropout(x))


class DownstreamClassifier(nn.Module):
    """Wrapper: backbone PatchTSTPhm + pooling + cabeza de clasificacion.

    Args:
        backbone: instancia de `models.patchtst_phm.PatchTSTPhm`.
        n_classes: numero de clases del downstream.
        freeze_backbone: si True, los parametros del backbone se ponen en
            `requires_grad=False` (linear probing). Por defecto False
            (from_scratch o full_finetuning).
        head_dropout: dropout opcional en la cabeza lineal.

    Forward:
        x:                       `(B, C, N, P)` patches.
        valid_time_mask:         `(B, W)` bool.
        valid_patch_mask:        canonicalizable a `(B, C, N)`.
        canales_constantes_mask: `(B, C)` bool opcional.

    Returns:
        Dict con keys:
            - `logits`: `(B, n_classes)`.
            - `pooled`: `(B, d_model)` (util para debugging).
            - `tokens`: `(B, C, N, d_model)` salida del backbone.

    Nota: la cabeza es la unica capa que depende del numero de clases del
    downstream. El backbone es completamente reutilizable entre datasets.
    """

    def __init__(
        self,
        backbone: nn.Module,
        n_classes: int,
        freeze_backbone: bool = False,
        head_dropout: float = 0.0,
    ):
        super().__init__()
        self.backbone = backbone
        self.head = ClassificationHead(
            d_model=getattr(backbone, "d_model"),
            n_classes=n_classes,
            dropout=head_dropout,
        )
        self.freeze_backbone = bool(freeze_backbone)
        if self.freeze_backbone:
            for p in self.backbone.parameters():
                p.requires_grad = False

    def trainable_parameter_groups(
        self,
        lr_head: float,
        lr_backbone: Optional[float] = None,
    ):
        """Devuelve grupos de parametros para AdamW con LRs distintos.

        En linear probing solo la cabeza entrena: el grupo del backbone
        queda vacio o ausente.

        En from_scratch / full_finetuning, si `lr_backbone` no se pasa,
        se usa `lr_head` para todo el modelo. Si se pasa, se crean dos
        grupos: backbone con `lr_backbone` y cabeza con `lr_head`.
        """
        head_params = list(self.head.parameters())
        if self.freeze_backbone:
            return [{"params": head_params, "lr": lr_head}]
        backbone_params = [p for p in self.backbone.parameters() if p.requires_grad]
        if lr_backbone is None or float(lr_backbone) == float(lr_head):
            return [{"params": head_params + backbone_params, "lr": lr_head}]
        return [
            {"params": backbone_params, "lr": lr_backbone},
            {"params": head_params,     "lr": lr_head},
        ]

    def forward(
        self,
        x: torch.Tensor,
        valid_time_mask: torch.Tensor,
        valid_patch_mask: torch.Tensor,
        canales_constantes_mask: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        # `ssl_mask=None` en downstream: no enmascaramos patches.
        out = self.backbone(x, valid_time_mask, valid_patch_mask, ssl_mask=None)
        tokens = out["tokens"]  # (B, C, N, d_model)
        # Reconstruir valid_patch_mask canonicalizado en caso de que llegue
        # con shape no canonica.
        from training.ssl.masking import canonicalize_valid_patch_mask
        B, C, N, d = tokens.shape
        vpm = canonicalize_valid_patch_mask(valid_patch_mask, B, C, N)
        pooled = pooled_embedding(tokens, vpm, canales_constantes_mask)
        logits = self.head(pooled)
        return {"logits": logits, "pooled": pooled, "tokens": tokens}


# ----------------------------------------------------------------------
# RegressionHead + RegressionDownstreamModel (commit 4 del bloque RUL)
# ----------------------------------------------------------------------


class RegressionHead(nn.Module):
    """Cabeza de regresion escalar sobre el embedding pooled.

    Salida por defecto: tensor `(B,)` (escalar por sample). Asi encaja
    directamente con `nn.MSELoss(reduction='mean')` y con metricas tipo
    MAE/RMSE sin necesidad de `squeeze` en el trainer. Si el caller
    necesita la dimension explicita `(B, 1)`, basta con instanciar la
    cabeza con `keep_last_dim=True` (util para stacking de tareas).

    Soporta dos arquitecturas:

      - **Lineal minima** (`hidden_dim=None`): solo `Linear(d_model, 1)`
        con dropout opcional aplicado al input. Es el default y se
        corresponde con linear probing puro.
      - **MLP de 2 capas** (`hidden_dim > 0`): `Linear(d_model, hidden)
        -> activation -> Dropout -> Linear(hidden, 1)`. La activacion
        intermedia se elige con `activation in {None, 'relu', 'gelu',
        'tanh'}`.

    Args:
        d_model: dimension del embedding de entrada (= `backbone.d_model`).
        hidden_dim: si None, cabeza lineal; si int > 0, MLP de 2 capas.
        dropout: probabilidad de dropout. En la version lineal se aplica
            antes del `Linear`. En la version MLP se aplica entre la
            activacion y la segunda `Linear`. Debe estar en `[0, 1)`.
        activation: nombre de la activacion intermedia para la version MLP.
            Solo se usa si `hidden_dim is not None`. Valores aceptados:
            `None`, `"none"`, `"relu"`, `"gelu"`, `"tanh"`.
        keep_last_dim: si True, la salida es `(B, 1)`. Por defecto `(B,)`.

    Raises:
        ValueError si `d_model <= 0`, `hidden_dim <= 0`, dropout fuera
        de `[0, 1)` o activation desconocida.
    """

    _ACTIVATIONS = {
        None: None,
        "none": None,
        "relu": nn.ReLU,
        "gelu": nn.GELU,
        "tanh": nn.Tanh,
    }

    def __init__(
        self,
        d_model: int,
        hidden_dim: Optional[int] = None,
        dropout: float = 0.0,
        activation: Optional[str] = None,
        keep_last_dim: bool = False,
    ):
        super().__init__()
        if d_model <= 0:
            raise ValueError(f"d_model debe ser > 0, recibido {d_model}")
        dropout_f = float(dropout)
        if not (0.0 <= dropout_f < 1.0):
            raise ValueError(
                f"dropout debe estar en [0, 1); recibido {dropout_f}"
            )
        if hidden_dim is not None and hidden_dim <= 0:
            raise ValueError(
                f"hidden_dim debe ser > 0 o None; recibido {hidden_dim}"
            )
        if activation not in self._ACTIVATIONS:
            raise ValueError(
                f"activation desconocida: {activation!r}. "
                f"Esperado uno de {list(self._ACTIVATIONS.keys())}."
            )
        self.d_model = int(d_model)
        self.hidden_dim = hidden_dim
        self.keep_last_dim = bool(keep_last_dim)
        self.is_mlp = hidden_dim is not None

        if not self.is_mlp:
            # Lineal minima: dropout en input + Linear(d_model, 1).
            self.dropout = nn.Dropout(dropout_f) if dropout_f > 0 else nn.Identity()
            self.fc = nn.Linear(self.d_model, 1)
        else:
            # MLP de 2 capas con activacion + dropout opcionales.
            act_cls = self._ACTIVATIONS[activation]
            layers: list = [nn.Linear(self.d_model, int(hidden_dim))]
            if act_cls is not None:
                layers.append(act_cls())
            if dropout_f > 0:
                layers.append(nn.Dropout(dropout_f))
            layers.append(nn.Linear(int(hidden_dim), 1))
            self.mlp = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Args:
            x: `(B, d_model)` embedding pooled.
        Returns:
            `(B,)` por defecto, o `(B, 1)` si `keep_last_dim=True`.
        """
        if x.dim() != 2:
            raise ValueError(
                f"x debe ser (B, d_model); recibido {tuple(x.shape)}"
            )
        if x.shape[1] != self.d_model:
            raise ValueError(
                f"x.shape[1]={x.shape[1]} != d_model={self.d_model}"
            )
        if self.is_mlp:
            out = self.mlp(x)        # (B, 1)
        else:
            out = self.fc(self.dropout(x))  # (B, 1)
        if self.keep_last_dim:
            return out
        return out.squeeze(-1)       # (B,)


class RegressionDownstreamModel(nn.Module):
    """Wrapper: backbone PatchTSTPhm + pooling + RegressionHead.

    Mismo contrato y semantica que `DownstreamClassifier`, adaptado a
    regresion escalar. La cabeza NO depende de `n_classes` (no aplica),
    solo de `d_model`.

    Forward:
        x:                       `(B, C, N, P)` patches.
        valid_time_mask:         `(B, W)` bool, W = N*P.
        valid_patch_mask:        canonicalizable a `(B, C, N)`.
        canales_constantes_mask: `(B, C)` bool opcional.

    Returns:
        Dict con keys:
            - `prediction`: `(B,)` por defecto, o `(B, 1)` si la cabeza
              se configuro con `keep_last_dim=True`.
            - `pooled`:     `(B, d_model)` (util para diagnostico).
            - `tokens`:     `(B, C, N, d_model)` salida del backbone.

    Args:
        backbone: instancia de `models.patchtst_phm.PatchTSTPhm`.
        freeze_backbone: si True, congela el backbone (linear probing).
        head_hidden_dim, head_dropout, head_activation, head_keep_last_dim:
            forwarded a `RegressionHead`.
    """

    def __init__(
        self,
        backbone: nn.Module,
        freeze_backbone: bool = False,
        head_hidden_dim: Optional[int] = None,
        head_dropout: float = 0.0,
        head_activation: Optional[str] = None,
        head_keep_last_dim: bool = False,
    ):
        super().__init__()
        self.backbone = backbone
        self.head = RegressionHead(
            d_model=getattr(backbone, "d_model"),
            hidden_dim=head_hidden_dim,
            dropout=head_dropout,
            activation=head_activation,
            keep_last_dim=head_keep_last_dim,
        )
        self.freeze_backbone = bool(freeze_backbone)
        if self.freeze_backbone:
            for p in self.backbone.parameters():
                p.requires_grad = False

    def trainable_parameter_groups(
        self,
        lr_head: float,
        lr_backbone: Optional[float] = None,
    ):
        """Devuelve grupos de parametros para AdamW con LRs distintos.

        Semantica idéntica a `DownstreamClassifier.trainable_parameter_groups`:

        - `freeze_backbone=True`  -> solo cabeza con `lr_head`.
        - `lr_backbone` ausente o igual a `lr_head` -> un solo grupo.
        - `lr_backbone` distinto -> dos grupos (backbone con `lr_backbone`,
          cabeza con `lr_head`).
        """
        head_params = list(self.head.parameters())
        if self.freeze_backbone:
            return [{"params": head_params, "lr": lr_head}]
        backbone_params = [p for p in self.backbone.parameters() if p.requires_grad]
        if lr_backbone is None or float(lr_backbone) == float(lr_head):
            return [{"params": head_params + backbone_params, "lr": lr_head}]
        return [
            {"params": backbone_params, "lr": lr_backbone},
            {"params": head_params,     "lr": lr_head},
        ]

    def forward(
        self,
        x: torch.Tensor,
        valid_time_mask: torch.Tensor,
        valid_patch_mask: torch.Tensor,
        canales_constantes_mask: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        out = self.backbone(x, valid_time_mask, valid_patch_mask, ssl_mask=None)
        tokens = out["tokens"]  # (B, C, N, d_model)
        # Canonicalizar valid_patch_mask por si llega (B, N) o (B, C, N).
        from training.ssl.masking import canonicalize_valid_patch_mask
        B, C, N, d = tokens.shape
        vpm = canonicalize_valid_patch_mask(valid_patch_mask, B, C, N)
        pooled = pooled_embedding(tokens, vpm, canales_constantes_mask)
        prediction = self.head(pooled)
        return {"prediction": prediction, "pooled": pooled, "tokens": tokens}
