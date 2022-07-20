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
        self.relu25 = ReLU(inplace=True)

    def forward(self, x97, x92):
        x98=operator.add(x97, x92)
        x99=self.relu25(x98)
        return x99

m = M().eval()
x97 = torch.randn(torch.Size([1, 256, 14, 14]))
x92 = torch.randn(torch.Size([1, 256, 14, 14]))
start = time.time()
output = m(x97, x92)
end = time.time()
print(end-start)
