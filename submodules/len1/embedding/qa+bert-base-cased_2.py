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
        self.embedding2 = Embedding(512, 768)
        self._tensor_constant580 = torch.rand(torch.Size([1, 384])).to(torch.int64)

    def forward(self, ):
        x25=self.embedding2(self._tensor_constant580)
        return x25

m = M().eval()
start = time.time()
output = m()
end = time.time()
print(end-start)
