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
        self.layer_scale3 = torch.rand(torch.Size([192, 1, 1])).to(torch.float32)

    def forward(self, x53):
        x54=operator.mul(self.layer_scale3, x53)
        return x54

m = M().eval()
x53 = torch.randn(torch.Size([1, 192, 28, 28]))
start = time.time()
output = m(x53)
end = time.time()
print(end-start)
