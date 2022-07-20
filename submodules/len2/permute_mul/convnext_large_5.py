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
        self.layer_scale5 = torch.rand(torch.Size([384, 1, 1]))

    def forward(self, x74):
        x75=torch.permute(x74, [0, 3, 1, 2])
        x76=operator.mul(self.layer_scale5, x75)
        return x76

m = M().eval()
x74 = torch.randn(torch.Size([1, 28, 28, 384]))
start = time.time()
output = m(x74)
end = time.time()
print(end-start)
