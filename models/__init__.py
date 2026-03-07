"""Modelos del proyecto fm_fl_phmd.

Por ahora solo `PatchTSTPhm`: encoder PatchTST channel-independent adaptado
al contrato PHM (W=512, P=16, N=32).
"""

from models.patchtst_phm import PatchTSTPhm, build_patchtst_phm, count_parameters

__all__ = ["PatchTSTPhm", "build_patchtst_phm", "count_parameters"]
