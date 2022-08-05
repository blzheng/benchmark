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
        self.layer_scale25 = torch.rand(torch.Size([384, 1, 1])).to(torch.float32)

    def forward(self, x301):
        x302=operator.mul(self.layer_scale25, x301)
        return x302

m = M().eval()
x301 = torch.randn(torch.Size([1, 384, 14, 14]))
start = time.time()
output = m(x301)
end = time.time()
print(end-start)
