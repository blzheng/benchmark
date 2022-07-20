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
        self.sigmoid11 = Sigmoid()

    def forward(self, x247, x243):
        x248=self.sigmoid11(x247)
        x249=operator.mul(x248, x243)
        return x249

m = M().eval()
x247 = torch.randn(torch.Size([1, 960, 1, 1]))
x243 = torch.randn(torch.Size([1, 960, 14, 14]))
start = time.time()
output = m(x247, x243)
end = time.time()
print(end-start)
