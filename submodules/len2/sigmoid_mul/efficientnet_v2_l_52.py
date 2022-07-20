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
        self.sigmoid52 = Sigmoid()

    def forward(self, x953, x949):
        x954=self.sigmoid52(x953)
        x955=operator.mul(x954, x949)
        return x955

m = M().eval()
x953 = torch.randn(torch.Size([1, 2304, 1, 1]))
x949 = torch.randn(torch.Size([1, 2304, 7, 7]))
start = time.time()
output = m(x953, x949)
end = time.time()
print(end-start)
