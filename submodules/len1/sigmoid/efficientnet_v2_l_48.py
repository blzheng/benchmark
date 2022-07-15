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
        self.sigmoid48 = Sigmoid()

    def forward(self, x889):
        x890=self.sigmoid48(x889)
        return x890

m = M().eval()
x889 = torch.randn(torch.Size([1, 2304, 1, 1]))
start = time.time()
output = m(x889)
end = time.time()
print(end-start)
