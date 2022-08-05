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
        self.layer_scale7 = torch.rand(torch.Size([768, 1, 1])).to(torch.float32)

    def forward(self, x102):
        x103=torch.permute(x102, [0, 3, 1, 2])
        x104=operator.mul(self.layer_scale7, x103)
        return x104

m = M().eval()
x102 = torch.randn(torch.Size([1, 14, 14, 768]))
start = time.time()
output = m(x102)
end = time.time()
print(end-start)
