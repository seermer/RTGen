import torch
import torch.nn as nn
from .builder import QUEUE
from collections import deque


@QUEUE.register_module()
class GeneratedQueue(nn.Module):
    def __init__(self, name, length, emb_dim=512):
        super(GeneratedQueue, self).__init__()
        self.name = name
        self.emb_dim = emb_dim
        self.queue = deque(maxlen=length)
        self._param = torch.nn.Parameter(torch.zeros(1, emb_dim),
                                         requires_grad=False)

    @torch.no_grad()
    def dequeue_and_enqueue(self, queue_update):
        embs = queue_update[self.name]
        for emb in embs.detach():
            self.queue.append(emb)

    @torch.no_grad()
    def get_queue(self, key):
        assert key == self.name
        if len(self.queue) == 0:
            return torch.zeros((0, self.emb_dim), device=self._param.device)
        return torch.stack(list(self.queue))


