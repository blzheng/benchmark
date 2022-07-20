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
        self.relu57 = ReLU(inplace=True)

    def forward(self, x189, x197):
        x198=operator.add(x189, x197)
        x199=self.relu57(x198)
        return x199

m = M().eval()
x189 = torch.randn(torch.Size([1, 1344, 14, 14]))
x197 = torch.randn(torch.Size([1, 1344, 14, 14]))
start = time.time()
output = m(x189, x197)
end = time.time()
print(end-start)
