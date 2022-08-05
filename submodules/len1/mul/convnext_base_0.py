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
        self.layer_scale0 = torch.rand(torch.Size([128, 1, 1])).to(torch.float32)

    def forward(self, x14):
        x15=operator.mul(self.layer_scale0, x14)
        return x15

m = M().eval()
x14 = torch.randn(torch.Size([1, 128, 56, 56]))
start = time.time()
output = m(x14)
end = time.time()
print(end-start)
