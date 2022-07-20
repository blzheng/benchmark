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

    def forward(self, x64):
        x65=operator.mul(self.layer_scale4, x64)
        return x65

m = M().eval()
x64 = torch.randn(torch.Size([1, 192, 28, 28]))
start = time.time()
output = m(x64)
end = time.time()
print(end-start)