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
        self.dropout33 = Dropout(p=0.1, inplace=False)

    def forward(self, x487, x484):
        x488=self.dropout33(x487)
        x489=operator.add(x488, x484)
        return x489

m = M().eval()
x487 = torch.randn(torch.Size([1, 384, 768]))
x484 = torch.randn(torch.Size([1, 384, 768]))
start = time.time()
output = m(x487, x484)
end = time.time()
print(end-start)
