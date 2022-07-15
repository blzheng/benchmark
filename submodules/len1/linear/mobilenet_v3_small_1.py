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
        self.linear1 = Linear(in_features=1024, out_features=1000, bias=True)

    def forward(self, x156):
        x157=self.linear1(x156)
        return x157

m = M().eval()
x156 = torch.randn(torch.Size([1, 1024]))
start = time.time()
output = m(x156)
end = time.time()
print(end-start)
