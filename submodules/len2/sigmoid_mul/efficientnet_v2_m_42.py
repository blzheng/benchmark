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
        self.sigmoid42 = Sigmoid()

    def forward(self, x758, x754):
        x759=self.sigmoid42(x758)
        x760=operator.mul(x759, x754)
        return x760

m = M().eval()
x758 = torch.randn(torch.Size([1, 3072, 1, 1]))
x754 = torch.randn(torch.Size([1, 3072, 7, 7]))
start = time.time()
output = m(x758, x754)
end = time.time()
print(end-start)
