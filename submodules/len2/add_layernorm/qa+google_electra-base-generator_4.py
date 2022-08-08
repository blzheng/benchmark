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
        self.layernorm4 = LayerNorm((256,), eps=1e-12, elementwise_affine=True)

    def forward(self, x111, x107):
        x112=operator.add(x111, x107)
        x113=self.layernorm4(x112)
        return x113

m = M().eval()
x111 = torch.randn(torch.Size([1, 384, 256]))
x107 = torch.randn(torch.Size([1, 384, 256]))
start = time.time()
output = m(x111, x107)
end = time.time()
print(end-start)
