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
        self.dropout5 = Dropout(p=0.1, inplace=False)
        self.layernorm3 = LayerNorm((768,), eps=1e-05, elementwise_affine=True)

    def forward(self, x103, x70):
        x104=self.dropout5(x103)
        x105=operator.add(x104, x70)
        x106=self.layernorm3(x105)
        return x106

m = M().eval()
x103 = torch.randn(torch.Size([1, 384, 768]))
x70 = torch.randn(torch.Size([1, 384, 768]))
start = time.time()
output = m(x103, x70)
end = time.time()
print(end-start)
