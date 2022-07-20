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
        self.layernorm34 = LayerNorm((1024,), eps=1e-06, elementwise_affine=True)

    def forward(self, x401):
        x402=self.layernorm34(x401)
        return x402

m = M().eval()
x401 = torch.randn(torch.Size([1, 7, 7, 1024]))
start = time.time()
output = m(x401)
end = time.time()
print(end-start)
