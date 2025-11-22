import torch
import torch.nn as nn
from typing import Dict, Tuple

class AttentionBus:
    """
    Lightweight key/value bus for experts to publish and read each other's summaries.

    Usage:
      bus = AttentionBus(d_model=128)
      bus.publish('vision', k, v)
      aggregated = bus.read_and_attend(query, topk=4)
    """
    def __init__(self, d_model: int = 128, device: str = 'cpu'):
        self.device = device
        self.d_model = d_model
        self.store: Dict[str, Tuple[torch.Tensor, torch.Tensor]] = {}  # name -> (k, v)

    def publish(self, name: str, k: torch.Tensor, v: torch.Tensor):
        """Publish key k [N_k, D], value v [N_k, D_v] under name."""
        # Ensure tensors are on bus device
        k = k.to(self.device)
        v = v.to(self.device)
        self.store[name] = (k, v)

    def clear(self):
        self.store.clear()

    def read_and_attend(self, query: torch.Tensor, softmax_temp: float = 1.0) -> torch.Tensor:
        """
        Concatenate all keys/values and attend from query.
        query: [Q, D]
        Returns: context [Q, D_v]
        """
        if not self.store:
            # empty
            return torch.zeros(query.shape[0], self.d_model, device=query.device)

        all_k = []
        all_v = []
        for k, v in self.store.values():
            all_k.append(k)
            all_v.append(v)
        K = torch.cat(all_k, dim=0)  # [M, D]
        V = torch.cat(all_v, dim=0)  # [M, D_v]

        # Simple scaled dot-product attention
        # query [Q, D], K [M, D]
        scores = torch.matmul(query, K.t()) / (self.d_model ** 0.5)
        weights = torch.softmax(scores / softmax_temp, dim=-1)
        context = torch.matmul(weights, V)
        return context
