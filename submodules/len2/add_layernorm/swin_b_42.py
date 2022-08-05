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
        self.layernorm50 = LayerNorm((1024,), eps=1e-05, elementwise_affine=True)

    def forward(self, x548, x555):
        x556=operator.add(x548, x555)
        x557=self.layernorm50(x556)
        return x557

m = M().eval()
x548 = torch.randn(torch.Size([1, 7, 7, 1024]))
x555 = torch.randn(torch.Size([1, 7, 7, 1024]))
start = time.time()
output = m(x548, x555)
end = time.time()
print(end-start)
