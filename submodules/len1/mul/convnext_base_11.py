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
        self.layer_scale11 = torch.rand(torch.Size([512, 1, 1])).to(torch.float32)

    def forward(self, x147):
        x148=operator.mul(self.layer_scale11, x147)
        return x148

m = M().eval()
x147 = torch.randn(torch.Size([1, 512, 14, 14]))
start = time.time()
output = m(x147)
end = time.time()
print(end-start)
