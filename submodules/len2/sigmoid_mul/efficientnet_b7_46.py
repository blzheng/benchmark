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
        self.sigmoid46 = Sigmoid()

    def forward(self, x725, x721):
        x726=self.sigmoid46(x725)
        x727=operator.mul(x726, x721)
        return x727

m = M().eval()
x725 = torch.randn(torch.Size([1, 2304, 1, 1]))
x721 = torch.randn(torch.Size([1, 2304, 7, 7]))
start = time.time()
output = m(x725, x721)
end = time.time()
print(end-start)
