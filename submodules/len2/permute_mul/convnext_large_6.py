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
        self.layer_scale6 = torch.rand(torch.Size([768, 1, 1])).to(torch.float32)

    def forward(self, x91):
        x92=torch.permute(x91, [0, 3, 1, 2])
        x93=operator.mul(self.layer_scale6, x92)
        return x93

m = M().eval()
x91 = torch.randn(torch.Size([1, 14, 14, 768]))
start = time.time()
output = m(x91)
end = time.time()
print(end-start)
