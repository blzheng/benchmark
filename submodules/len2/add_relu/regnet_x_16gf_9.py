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
        self.relu30 = ReLU(inplace=True)

    def forward(self, x99, x107):
        x108=operator.add(x99, x107)
        x109=self.relu30(x108)
        return x109

m = M().eval()
x99 = torch.randn(torch.Size([1, 896, 14, 14]))
x107 = torch.randn(torch.Size([1, 896, 14, 14]))
start = time.time()
output = m(x99, x107)
end = time.time()
print(end-start)
