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
        self.layer_scale13 = torch.rand(torch.Size([768, 1, 1])).to(torch.float32)

    def forward(self, x169):
        x170=operator.mul(self.layer_scale13, x169)
        return x170

m = M().eval()
x169 = torch.randn(torch.Size([1, 768, 14, 14]))
start = time.time()
output = m(x169)
end = time.time()
print(end-start)
