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
        self.layer_scale16 = torch.rand(torch.Size([768, 1, 1]))

    def forward(self, x201):
        x202=torch.permute(x201, [0, 3, 1, 2])
        x203=operator.mul(self.layer_scale16, x202)
        return x203

m = M().eval()
x201 = torch.randn(torch.Size([1, 14, 14, 768]))
start = time.time()
output = m(x201)
end = time.time()
print(end-start)
