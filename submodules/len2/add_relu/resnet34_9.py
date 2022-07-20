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
        self.relu19 = ReLU(inplace=True)

    def forward(self, x76, x71):
        x77=operator.add(x76, x71)
        x78=self.relu19(x77)
        return x78

m = M().eval()
x76 = torch.randn(torch.Size([1, 256, 14, 14]))
x71 = torch.randn(torch.Size([1, 256, 14, 14]))
start = time.time()
output = m(x76, x71)
end = time.time()
print(end-start)
