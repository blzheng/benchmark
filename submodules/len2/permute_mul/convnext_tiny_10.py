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
        self.layer_scale10 = torch.rand(torch.Size([384, 1, 1]))

    def forward(self, x135):
        x136=torch.permute(x135, [0, 3, 1, 2])
        x137=operator.mul(self.layer_scale10, x136)
        return x137

m = M().eval()
x135 = torch.randn(torch.Size([1, 14, 14, 384]))
start = time.time()
output = m(x135)
end = time.time()
print(end-start)
