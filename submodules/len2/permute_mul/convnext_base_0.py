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
        self.layer_scale0 = torch.rand(torch.Size([128, 1, 1]))

    def forward(self, x13):
        x14=torch.permute(x13, [0, 3, 1, 2])
        x15=operator.mul(self.layer_scale0, x14)
        return x15

m = M().eval()
x13 = torch.randn(torch.Size([1, 56, 56, 128]))
start = time.time()
output = m(x13)
end = time.time()
print(end-start)
