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
        self.sigmoid16 = Sigmoid()

    def forward(self, x277, x273):
        x278=self.sigmoid16(x277)
        x279=operator.mul(x278, x273)
        return x279

m = M().eval()
x277 = torch.randn(torch.Size([1, 2016, 1, 1]))
x273 = torch.randn(torch.Size([1, 2016, 7, 7]))
start = time.time()
output = m(x277, x273)
end = time.time()
print(end-start)
