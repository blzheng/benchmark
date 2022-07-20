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
        self.relu48 = ReLU(inplace=True)

    def forward(self, x159, x167):
        x168=operator.add(x159, x167)
        x169=self.relu48(x168)
        return x169

m = M().eval()
x159 = torch.randn(torch.Size([1, 432, 14, 14]))
x167 = torch.randn(torch.Size([1, 432, 14, 14]))
start = time.time()
output = m(x159, x167)
end = time.time()
print(end-start)
