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
        self.layer_scale3 = torch.rand(torch.Size([384, 1, 1]))

    def forward(self, x52):
        x53=torch.permute(x52, [0, 3, 1, 2])
        x54=operator.mul(self.layer_scale3, x53)
        return x54

m = M().eval()
x52 = torch.randn(torch.Size([1, 28, 28, 384]))
start = time.time()
output = m(x52)
end = time.time()
print(end-start)
