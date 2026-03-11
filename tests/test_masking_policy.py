"""Tests de la politica de masking para SSL (`training.ssl.masking`).

Cubre las garantias canonicas:

- `ssl_mask` nunca selecciona patches invalidos.
- `effective_mask_ratio` se acerca al objetivo cuando hay suficientes patches.
- Casos con pocos patches validos no rompen.
- `min_masks` se aplica correctamente.
- `canonicalize_valid_patch_mask` soporta todas las shapes documentadas.
"""

from __future__ import annotations

import pytest
import torch

from training.ssl.masking import (
    canonicalize_valid_patch_mask,
    compute_effective_mask_ratio,
    generate_ssl_mask,
)


# ----------------------------------------------------------------------
# canonicalize_valid_patch_mask
# ----------------------------------------------------------------------


@pytest.mark.parametrize(
    "shape_in",
    [(2, 4, 8), (2, 1, 8), (2, 8), (4, 8), (8,)],
)
def test_canonicalize_supports_all_shapes(shape_in):
    B, C, N = 2, 4, 8
    m = torch.ones(shape_in, dtype=torch.bool)
    out = canonicalize_valid_patch_mask(m, B, C, N)
    assert out.shape == (B, C, N)
    assert out.dtype == torch.bool
    assert out.all().item()


def test_canonicalize_rejects_bad_shape():
    with pytest.raises(ValueError, match="incompatible"):
        canonicalize_valid_patch_mask(
            torch.ones(3, 9, dtype=torch.bool), B=2, C=4, N=8
        )


def test_canonicalize_rejects_non_bool():
    with pytest.raises(ValueError, match="bool"):
        canonicalize_valid_patch_mask(
            torch.ones(2, 4, 8), B=2, C=4, N=8
        )


# ----------------------------------------------------------------------
# generate_ssl_mask: nunca selecciona invalidos
# ----------------------------------------------------------------------


def test_ssl_mask_never_selects_invalid():
    """Para cualquier configuracion de validez, ssl_mask ⊆ valid_patch_mask."""
    torch.manual_seed(0)
    B, C, N = 4, 3, 32
    vpm = torch.rand(B, C, N) > 0.4  # mezcla aleatoria
    ssl_mask = generate_ssl_mask(vpm, mask_ratio=0.3)
    # Estrictamente: ssl_mask & ~vpm debe ser todo False
    assert not (ssl_mask & ~vpm).any().item()


def test_ssl_mask_no_selection_on_zero_valid_row():
    """Filas (b,c) sin validos no deben tener ssl_mask=True en ninguna posicion."""
    B, C, N = 2, 2, 16
    vpm = torch.ones(B, C, N, dtype=torch.bool)
    vpm[0, 0, :] = False
    ssl_mask = generate_ssl_mask(vpm, mask_ratio=0.5)
    assert not ssl_mask[0, 0].any().item()


# ----------------------------------------------------------------------
# generate_ssl_mask: effective_mask_ratio
# ----------------------------------------------------------------------


def test_effective_mask_ratio_close_to_target_when_many_valid():
    """Con muchos patches validos, el ratio efectivo aproxima al objetivo."""
    torch.manual_seed(1)
    B, C, N = 8, 4, 32
    vpm = torch.ones(B, C, N, dtype=torch.bool)
    target = 0.3
    ssl_mask = generate_ssl_mask(vpm, mask_ratio=target)
    eff = compute_effective_mask_ratio(ssl_mask, vpm)
    # Tolerancia: con N=32, round(0.3*32)=10/32=0.3125. Damos margen.
    assert abs(eff - target) < 0.05, f"eff={eff:.4f}, target={target}"


# ----------------------------------------------------------------------
# generate_ssl_mask: min_masks
# ----------------------------------------------------------------------


def test_min_masks_applied_when_valid_count_low():
    """Con n_valid >= 1 y mask_ratio bajo, el min_masks se aplica."""
    B, C, N = 1, 1, 4
    vpm = torch.zeros(B, C, N, dtype=torch.bool)
    vpm[0, 0, :2] = True  # 2 patches validos
    # Con mask_ratio=0.1, round(0.1*2)=0, pero min_masks=1 fuerza 1
    ssl_mask = generate_ssl_mask(vpm, mask_ratio=0.1, min_masks=1)
    assert int(ssl_mask.sum().item()) == 1


def test_min_masks_clipped_by_valid_count():
    """Si min_masks > n_valid, se selecciona como mucho n_valid."""
    B, C, N = 1, 1, 4
    vpm = torch.zeros(B, C, N, dtype=torch.bool)
    vpm[0, 0, :2] = True
    ssl_mask = generate_ssl_mask(vpm, mask_ratio=0.5, min_masks=10)
    # No puede haber mas masks que validos
    assert int(ssl_mask.sum().item()) == 2
    # Y solo entre los validos
    assert not (ssl_mask & ~vpm).any().item()


# ----------------------------------------------------------------------
# generate_ssl_mask: shapes y errores
# ----------------------------------------------------------------------


def test_generate_rejects_non_3d():
    with pytest.raises(ValueError, match="\\(B, C, N\\)"):
        generate_ssl_mask(torch.zeros(8, dtype=torch.bool))


def test_generate_rejects_bad_ratio():
    vpm = torch.ones(1, 1, 8, dtype=torch.bool)
    with pytest.raises(ValueError, match="\\[0, 1\\]"):
        generate_ssl_mask(vpm, mask_ratio=1.5)
    with pytest.raises(ValueError, match="\\[0, 1\\]"):
        generate_ssl_mask(vpm, mask_ratio=-0.1)


# ----------------------------------------------------------------------
# generate_ssl_mask: reproducibilidad con generator
# ----------------------------------------------------------------------


def test_reproducible_with_generator():
    vpm = torch.ones(2, 3, 32, dtype=torch.bool)
    g1 = torch.Generator().manual_seed(7)
    g2 = torch.Generator().manual_seed(7)
    m1 = generate_ssl_mask(vpm, mask_ratio=0.3, generator=g1)
    m2 = generate_ssl_mask(vpm, mask_ratio=0.3, generator=g2)
    assert (m1 == m2).all().item()


# ----------------------------------------------------------------------
# apply_mask_token
# ----------------------------------------------------------------------


def test_apply_mask_token_replaces_only_masked_positions():
    from training.ssl.masking import apply_mask_token

    BC, N, d = 2, 4, 8
    emb = torch.randn(BC, N, d)
    mask_token = torch.full((d,), -7.0)
    ssl_flat = torch.zeros(BC, N, dtype=torch.bool)
    ssl_flat[0, 1] = True
    ssl_flat[1, 3] = True
    out = apply_mask_token(emb, ssl_flat, mask_token)
    # Las posiciones masked tienen valor del mask_token
    assert torch.allclose(out[0, 1], mask_token)
    assert torch.allclose(out[1, 3], mask_token)
    # Las demas conservan el embedding original
    keep = torch.ones_like(ssl_flat, dtype=torch.bool)
    keep[0, 1] = False
    keep[1, 3] = False
    assert torch.allclose(out[keep], emb[keep])
