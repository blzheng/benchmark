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
        self.relu112 = ReLU(inplace=True)

    def forward(self, x388, x380):
        x389=operator.add(x388, x380)
        x390=self.relu112(x389)
        return x390

m = M().eval()
x388 = torch.randn(torch.Size([1, 1024, 14, 14]))
x380 = torch.randn(torch.Size([1, 1024, 14, 14]))
start = time.time()
output = m(x388, x380)
end = time.time()
print(end-start)
