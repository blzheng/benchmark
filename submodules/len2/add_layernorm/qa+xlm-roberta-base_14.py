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
        self.layernorm14 = LayerNorm((768,), eps=1e-05, elementwise_affine=True)

    def forward(self, x320, x316):
        x321=operator.add(x320, x316)
        x322=self.layernorm14(x321)
        return x322

m = M().eval()
x320 = torch.randn(torch.Size([1, 384, 768]))
x316 = torch.randn(torch.Size([1, 384, 768]))
start = time.time()
output = m(x320, x316)
end = time.time()
print(end-start)
