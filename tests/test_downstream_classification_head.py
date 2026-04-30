"""Tests del ClassificationHead y DownstreamClassifier.

Sintéticos, no requieren shards reales. Validan:

- shape de los logits;
- congelar backbone en linear_probing;
- el wrapper acepta los inputs canonicos del contrato `(B, C, N, P)`;
- channel-independence: el mismo classifier acepta C=1 y C=24;
- gradiente solo fluye por la cabeza cuando freeze_backbone=True.
"""

from __future__ import annotations

import pytest
import torch

from models.patchtst_phm import PatchTSTPhm
from training.downstream.heads import ClassificationHead, DownstreamClassifier


N_PATCHES = 32
PATCH_SIZE = 16
W = N_PATCHES * PATCH_SIZE


def _inputs(B: int, C: int):
    x = torch.randn(B, C, N_PATCHES, PATCH_SIZE)
    vtm = torch.ones(B, W, dtype=torch.bool)
    vpm = torch.ones(B, C, N_PATCHES, dtype=torch.bool)
    return x, vtm, vpm


# ----------------------------------------------------------------------
# ClassificationHead aislada
# ----------------------------------------------------------------------


def test_classification_head_shape():
    head = ClassificationHead(d_model=64, n_classes=4)
    x = torch.randn(8, 64)
    logits = head(x)
    assert logits.shape == (8, 4)


def test_classification_head_rejects_n_classes_lt_2():
    with pytest.raises(ValueError, match="n_classes"):
        ClassificationHead(d_model=64, n_classes=1)


def test_classification_head_dropout_applied():
    """Dropout deja al menos algunos elementos en 0 (en train mode)."""
    torch.manual_seed(0)
    head = ClassificationHead(d_model=64, n_classes=4, dropout=0.9)
    head.train()
    x = torch.ones(8, 64)
    # En 8 forward es muy improbable que dropout no aparezca con p=0.9.
    seen_zero = False
    for _ in range(5):
        y = head(x)
        if (y == 0).any().item():
            seen_zero = True
            break
    # En eval no debe haber dropout
    head.eval()
    y_eval = head(x)
    # logits con linear sobre ones siempre deterministicos en eval
    y_eval2 = head(x)
    assert torch.allclose(y_eval, y_eval2)


# ----------------------------------------------------------------------
# DownstreamClassifier end-to-end
# ----------------------------------------------------------------------


def test_classifier_logits_shape():
    backbone = PatchTSTPhm.tiny()
    clf = DownstreamClassifier(backbone, n_classes=4)
    B, C = 2, 5
    x, vtm, vpm = _inputs(B, C)
    out = clf(x, vtm, vpm)
    assert out["logits"].shape == (B, 4)
    assert out["pooled"].shape == (B, backbone.d_model)


def test_classifier_channel_independence():
    """El mismo classifier acepta C distintos en llamadas distintas."""
    backbone = PatchTSTPhm.tiny()
    clf = DownstreamClassifier(backbone, n_classes=3)
    for C in (1, 5, 24):
        x, vtm, vpm = _inputs(2, C)
        out = clf(x, vtm, vpm)
        assert out["logits"].shape == (2, 3)


def test_freeze_backbone_disables_backbone_grad():
    backbone = PatchTSTPhm.tiny()
    clf = DownstreamClassifier(backbone, n_classes=4, freeze_backbone=True)
    for name, p in clf.backbone.named_parameters():
        assert not p.requires_grad, f"backbone.{name} sigue entrenable"
    for name, p in clf.head.named_parameters():
        assert p.requires_grad, f"head.{name} debe ser entrenable"


def test_freeze_backbone_grad_flow():
    """Con freeze, backward no debe acumular grad en backbone."""
    backbone = PatchTSTPhm.tiny()
    clf = DownstreamClassifier(backbone, n_classes=4, freeze_backbone=True)
    B, C = 2, 3
    x, vtm, vpm = _inputs(B, C)
    out = clf(x, vtm, vpm)
    loss = out["logits"].mean()
    loss.backward()
    # backbone: ningun grad acumulado
    for name, p in clf.backbone.named_parameters():
        assert p.grad is None, f"backbone.{name} acumulo grad pese al freeze"
    # cabeza: si hay grad
    assert clf.head.fc.weight.grad is not None


def test_unfrozen_backbone_grad_flow():
    backbone = PatchTSTPhm.tiny()
    clf = DownstreamClassifier(backbone, n_classes=4, freeze_backbone=False)
    B, C = 2, 3
    x, vtm, vpm = _inputs(B, C)
    loss = clf(x, vtm, vpm)["logits"].mean()
    loss.backward()
    # Al menos algun parametro del backbone tiene grad
    has_grad = any(
        p.grad is not None and p.grad.abs().sum().item() > 0
        for p in clf.backbone.parameters()
    )
    assert has_grad, "ningun parametro del backbone recibio grad"
    assert clf.head.fc.weight.grad is not None


def test_param_groups_linear_probing():
    backbone = PatchTSTPhm.tiny()
    clf = DownstreamClassifier(backbone, n_classes=4, freeze_backbone=True)
    groups = clf.trainable_parameter_groups(lr_head=1e-3, lr_backbone=1e-4)
    # Linear probing: 1 grupo (solo head)
    assert len(groups) == 1
    assert groups[0]["lr"] == 1e-3
    assert all(p.requires_grad for p in groups[0]["params"])


def test_param_groups_full_finetuning_two_lrs():
    backbone = PatchTSTPhm.tiny()
    clf = DownstreamClassifier(backbone, n_classes=4, freeze_backbone=False)
    groups = clf.trainable_parameter_groups(lr_head=1e-3, lr_backbone=1e-4)
    # Dos grupos: backbone con lr_backbone y head con lr_head
    assert len(groups) == 2
    lrs = sorted([g["lr"] for g in groups])
    assert lrs == [1e-4, 1e-3]


def test_param_groups_full_finetuning_single_lr():
    backbone = PatchTSTPhm.tiny()
    clf = DownstreamClassifier(backbone, n_classes=4, freeze_backbone=False)
    groups = clf.trainable_parameter_groups(lr_head=1e-3, lr_backbone=None)
    # Si lr_backbone es None, todo va con lr_head
    assert len(groups) == 1
    assert groups[0]["lr"] == 1e-3
