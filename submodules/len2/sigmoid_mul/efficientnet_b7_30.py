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
        self.sigmoid30 = Sigmoid()

    def forward(self, x471, x467):
        x472=self.sigmoid30(x471)
        x473=operator.mul(x472, x467)
        return x473

m = M().eval()
x471 = torch.randn(torch.Size([1, 1344, 1, 1]))
x467 = torch.randn(torch.Size([1, 1344, 14, 14]))
start = time.time()
output = m(x471, x467)
end = time.time()
print(end-start)
