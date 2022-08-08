import torch
from torch import tensor
import torch.nn as nn
from torch.nn import *
import torchvision
import torchvision.models as models
from torchvision.ops.stochastic_depth import stochastic_depth
import time
import builtins
import operator

class M(torch.nn.Module):
    def __init__(self):
        super(M, self).__init__()
        self.embedding0 = Embedding(28996, 768, padding_idx=0)
        self._tensor_constant00 = torch.rand(torch.Size([1, 384])).to(torch.int64)

    def forward(self, ):
        x20=self.embedding0(self._tensor_constant00)
        return x20

m = M().eval()
start = time.time()
output = m()
end = time.time()
print(end-start)
