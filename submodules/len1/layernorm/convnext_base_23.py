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
        self.layernorm23 = LayerNorm((512,), eps=1e-06, elementwise_affine=True)

    def forward(self, x274):
        x275=self.layernorm23(x274)
        return x275

m = M().eval()
x274 = torch.randn(torch.Size([1, 14, 14, 512]))
start = time.time()
output = m(x274)
end = time.time()
print(end-start)
