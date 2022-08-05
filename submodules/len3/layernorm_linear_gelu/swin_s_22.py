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
        self.layernorm49 = LayerNorm((768,), eps=1e-05, elementwise_affine=True)
        self.linear47 = Linear(in_features=768, out_features=3072, bias=True)
        self.gelu22 = GELU(approximate='none')

    def forward(self, x548):
        x549=self.layernorm49(x548)
        x550=self.linear47(x549)
        x551=self.gelu22(x550)
        return x551

m = M().eval()
x548 = torch.randn(torch.Size([1, 7, 7, 768]))
start = time.time()
output = m(x548)
end = time.time()
print(end-start)
