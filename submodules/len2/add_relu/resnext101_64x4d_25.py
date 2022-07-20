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
        self.relu76 = ReLU(inplace=True)

    def forward(self, x268, x260):
        x269=operator.add(x268, x260)
        x270=self.relu76(x269)
        return x270

m = M().eval()
x268 = torch.randn(torch.Size([1, 1024, 14, 14]))
x260 = torch.randn(torch.Size([1, 1024, 14, 14]))
start = time.time()
output = m(x268, x260)
end = time.time()
print(end-start)
