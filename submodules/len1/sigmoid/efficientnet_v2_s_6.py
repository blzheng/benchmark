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
        self.sigmoid6 = Sigmoid()

    def forward(self, x169):
        x170=self.sigmoid6(x169)
        return x170

m = M().eval()
x169 = torch.randn(torch.Size([1, 768, 1, 1]))
start = time.time()
output = m(x169)
end = time.time()
print(end-start)
