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
        self.sigmoid4 = Sigmoid()

    def forward(self, x81, x77):
        x82=self.sigmoid4(x81)
        x83=operator.mul(x82, x77)
        return x83

m = M().eval()
x81 = torch.randn(torch.Size([1, 696, 1, 1]))
x77 = torch.randn(torch.Size([1, 696, 28, 28]))
start = time.time()
output = m(x81, x77)
end = time.time()
print(end-start)
