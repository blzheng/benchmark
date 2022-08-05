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
        self.layer_scale5 = torch.rand(torch.Size([256, 1, 1])).to(torch.float32)

    def forward(self, x75):
        x76=operator.mul(self.layer_scale5, x75)
        return x76

m = M().eval()
x75 = torch.randn(torch.Size([1, 256, 28, 28]))
start = time.time()
output = m(x75)
end = time.time()
print(end-start)
