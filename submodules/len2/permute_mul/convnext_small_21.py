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
        self.layer_scale21 = torch.rand(torch.Size([384, 1, 1])).to(torch.float32)

    def forward(self, x256):
        x257=torch.permute(x256, [0, 3, 1, 2])
        x258=operator.mul(self.layer_scale21, x257)
        return x258

m = M().eval()
x256 = torch.randn(torch.Size([1, 14, 14, 384]))
start = time.time()
output = m(x256)
end = time.time()
print(end-start)
