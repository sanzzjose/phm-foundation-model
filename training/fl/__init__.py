"""Federated learning simulado cross-silo para SSL pretraining.

Modulos:

- `aggregation`: `fedavg_state_dict(state_dicts, weights)` y utilidades
  numericas. Sin torch en el path simple (acepta tensores torch si los
  hay y los reduce con `torch.mean`; si no, hace fallback numpy).
- `client`: `FederatedClient` con `local_train(global_state_dict, ...)`.
  Reusa el backbone PatchTSTPhm, masking, loss y dataloader del SSL central.
  Cada cliente solo ve sus PRETRAIN_SOURCE.
- `server`: orquestador de rondas; agrega con FedAvg al final de cada
  ronda y loggea metricas (loss por cliente, drift, comunicacion).

API consumida por `training/train_ssl_federated.py`. Para el MVP el
algoritmo es FedAvg; FedProx queda como placeholder controlado por
`algorithm: fedavg | fedprox` en la config (no implementado todavia).
"""
